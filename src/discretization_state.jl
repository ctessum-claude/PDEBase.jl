struct EquationState <: AbstractEquationSystemDiscretization
    eqs::Vector{Equation}
    bceqs::Vector{Equation}
end

function EquationState()
    return EquationState(Equation[], Equation[])
end

function generate_system(
        disc_state::EquationState, s, u0, tspan, metadata,
        disc::AbstractEquationSystemDiscretization;
        checks=true
    )
    discvars = get_discvars(s)
    t = get_time(disc)
    name = getfield(metadata.pdesys, :name)
    pdesys = metadata.pdesys
    alleqs = vcat(disc_state.eqs, unique(disc_state.bceqs))
    alldepvarsdisc = vec(reduce(vcat, vec(unique(reduce(vcat, vec.(values(discvars)))))))

    # u0 is now stored in metadata (passed during metadata construction)
    # MTK v11's AtomicArrayDict doesn't allow indexed array symbolics as keys in initial_conditions
    # Only pass non-indexed initial conditions to System
    sys_defaults = Dict(pdesys.initial_conditions)

    ps_raw = get_ps(pdesys)
    ps_raw = ps_raw === nothing || ps_raw === SciMLBase.NullParameters() ? Num[] : ps_raw
    # get_ps may return Pairs (e.g. [v => 0.5]); extract symbols and merge values into defaults
    if !isempty(ps_raw) && first(ps_raw) isa Pair
        ps = Num[first(p) for p in ps_raw]
        merge!(sys_defaults, Dict(first(p) => last(p) for p in ps_raw))
    else
        ps = ps_raw
    end
    return try
        if t === nothing
            # At the time of writing, NonlinearProblems require that the system of equations be in this form:
            # 0 ~ ...
            # Thus, before creating a NonlinearSystem we normalize the equations s.t. the lhs is zero.
            eqs = map(eq -> 0 ~ eq.rhs - eq.lhs, alleqs)
            sys = System(
                eqs, alldepvarsdisc, ps, initial_conditions = sys_defaults, name = name,
                metadata = [ProblemTypeCtx => metadata], checks = checks
            )
            return sys, nothing
        else
            # * In the end we have reduced the problem to a system of equations in terms of Dt that can be solved by an ODE solver.

            sys = System(
                alleqs, t, alldepvarsdisc, ps; initial_conditions = sys_defaults, name = name,
                metadata = [ProblemTypeCtx => metadata], checks = checks
            )
            return sys, tspan
        end
    catch e
        println("The system of equations is:")
        println("Number of equations: ", length(alleqs))
        for (i, eq) in enumerate(alleqs)
            println("Eq $i: $eq")
        end
        println()
        println("Discretization failed, please post an issue on https://github.com/SciML/MethodOfLines.jl with the failing code and system at low point count.")
        println()
        rethrow(e)
    end
end

"""
    to_explicit_ode(sys)

Convert a time-dependent System from implicit ODE + algebraic form to explicit ODE form.
This is a lightweight alternative to `mtkcompile` for systems that are already structurally
simple (e.g., PDE discretizations from MethodOfLines).

Steps:
1. Identify algebraic equations (no time derivative): `u_k ~ expr`
2. Solve them for the constrained variable
3. Substitute into remaining ODE equations
4. Rearrange ODE equations from `D(u_k) + f(u) ~ 0` to `D(u_k) ~ -f(u)`
5. Return a new System without the algebraic variables
"""
function to_explicit_ode(sys)
    eqs = ModelingToolkit.equations(sys)
    dvs = ModelingToolkit.unknowns(sys)
    t = ModelingToolkit.get_iv(sys)
    D = Differential(t)

    # Classify equations: ODE vs algebraic
    ode_eqs = Equation[]
    algebraic_subs = Dict{Any,Any}()
    algebraic_dvs = Set{Any}()
    for eq in eqs
        # ArrayOp equations are already in explicit ODE form from MethodOfLines
        lhs_uw = unwrap(eq.lhs)
        if SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_uw))
            push!(ode_eqs, eq)
            continue
        end
        full_expr = eq.lhs - eq.rhs
        # Check if this equation has a time derivative
        has_deriv = false
        deriv_var = nothing
        for dv in dvs
            D_dv = unwrap(D(dv))
            if Symbolics.hasnode(x -> isequal(x, D_dv), unwrap(full_expr))
                has_deriv = true
                deriv_var = dv
                break
            end
        end
        if has_deriv
            push!(ode_eqs, eq)
        else
            # Algebraic equation: solve for the variable
            for dv in dvs
                dv_uw = unwrap(dv)
                if Symbolics.hasnode(x -> isequal(x, dv_uw), unwrap(full_expr))
                    # Simple case: u_k ~ expr or expr ~ u_k
                    rhs_val = solve_for(eq, dv)
                    algebraic_subs[dv] = rhs_val
                    push!(algebraic_dvs, dv)
                    break
                end
            end
        end
    end

    # Substitute algebraic variables into ODE equations
    if !isempty(algebraic_subs)
        ode_eqs = map(ode_eqs) do eq
            lhs = Symbolics.substitute(eq.lhs, algebraic_subs)
            rhs = Symbolics.substitute(eq.rhs, algebraic_subs)
            lhs ~ rhs
        end
    end

    # Rearrange ODE equations to explicit form: D(u_k) ~ rhs
    explicit_eqs = map(ode_eqs) do eq
        full_expr = eq.lhs - eq.rhs
        for dv in dvs
            dv in algebraic_dvs && continue
            D_dv = D(dv)
            D_dv_uw = unwrap(D_dv)
            if Symbolics.hasnode(x -> isequal(x, D_dv_uw), unwrap(full_expr))
                rhs_val = solve_for(full_expr ~ 0, D_dv)
                return D_dv ~ -rhs_val
            end
        end
        return eq  # fallback: return as-is
    end

    # Remove algebraic variables from unknowns
    remaining_dvs = filter(dv -> !(dv in algebraic_dvs), dvs)

    # Create observed equations for eliminated algebraic variables so they
    # remain evaluable (e.g., boundary variables u[1] = 0.0)
    obs_eqs = Equation[dv ~ algebraic_subs[dv] for dv in algebraic_dvs]

    # Reconstruct system preserving metadata
    ps = ModelingToolkit.parameters(sys)
    ic = ModelingToolkit.initial_conditions(sys)
    name = getfield(sys, :name)
    mol_metadata = getmetadata(sys, ProblemTypeCtx, nothing)
    meta = mol_metadata !== nothing ? [ProblemTypeCtx => mol_metadata] : nothing

    return System(explicit_eqs, t, remaining_dvs, ps;
                  observed=obs_eqs,
                  initial_conditions=ic, name=name, metadata=meta, checks=false)
end

function SciMLBase.discretize(
        pdesys::PDESystem,
        discretization::AbstractEquationSystemDiscretization;
        analytic = nothing, checks=true, simplify=true, kwargs...
    )
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization; checks=checks)
    return try
        simpsys = if simplify
            mtkcompile(sys)
        else
            complete(to_explicit_ode(sys))
        end
        if tspan === nothing
            add_metadata!(getmetadata(sys, ProblemTypeCtx, nothing), sys)
            # MTK v11 requires symbolic map for initial guess
            unknowns_list = ModelingToolkit.unknowns(simpsys)
            u0_guess = Dict(u => 1.0 for u in unknowns_list)
            return prob = NonlinearProblem(
                simpsys, u0_guess;
                discretization.kwargs..., kwargs...
            )
        else
            mol_metadata = getmetadata(simpsys, ProblemTypeCtx, nothing)
            add_metadata!(mol_metadata, sys)
            # Get u0 from metadata (stored there for MTK v11 compatibility)
            u0 = hasproperty(mol_metadata, :u0) ? mol_metadata.u0 : []
            # When using complete() instead of mtkcompile(), algebraic variables
            # (boundary conditions) have been removed from the system. Filter u0
            # to only include variables that are still unknowns.
            if !simplify
                reduced_dvs = Set(unwrap.(ModelingToolkit.unknowns(simpsys)))
                u0 = filter(p -> unwrap(first(p)) in reduced_dvs, u0)
            end
            # Get parameter values from the original pdesys initial_conditions
            # MTK v11 needs parameter values passed explicitly when creating ODEProblem
            pdesys_ic = mol_metadata.pdesys.initial_conditions
            ps_raw = get_ps(mol_metadata.pdesys)
            if ps_raw !== nothing && ps_raw !== SciMLBase.NullParameters() && !isempty(ps_raw)
                # get_ps may return Pairs (e.g. [v => 0.5]); extract parameter values
                param_vals = Dict{Any,Any}()
                if first(ps_raw) isa Pair
                    for p in ps_raw
                        param_vals[first(p)] = last(p)
                    end
                else
                    # Fall back to looking up parameters in initial_conditions
                    ps_unwrapped = [safe_unwrap(p) for p in ps_raw]
                    for (k, v) in pairs(pdesys_ic)
                        k_unwrapped = safe_unwrap(k)
                        if any(p -> isequal(k_unwrapped, safe_unwrap(p)), ps_unwrapped)
                            v_numeric = try
                                Symbolics.value(v)
                            catch
                                safe_unwrap(v)
                            end
                            param_vals[k] = v_numeric
                        end
                    end
                end
                if !isempty(param_vals)
                    # MTK v11 API: merge u0 and params into a single Dict
                    op = merge(Dict(u0), param_vals)
                    prob = ODEProblem(
                        simpsys, op, tspan; build_initializeprob = false,
                        allow_array_eqs = !simplify,
                        discretization.kwargs...,
                        kwargs...
                    )
                else
                    prob = ODEProblem(
                        simpsys, u0, tspan; build_initializeprob = false,
                        allow_array_eqs = !simplify,
                        discretization.kwargs...,
                        kwargs...
                    )
                end
            else
                prob = ODEProblem(
                    simpsys, u0, tspan; build_initializeprob = false,
                    discretization.kwargs...,
                    kwargs...
                )
            end
            if analytic === nothing
                return prob
            else
                f = ODEFunction(
                    pdesys, discretization, analytic = analytic,
                    discretization.kwargs..., kwargs...
                )

                return ODEProblem(
                    f, prob.u0, prob.tspan, prob.p;
                    discretization.kwargs..., kwargs...
                )
            end
        end
    catch e
        error_analysis(sys, e)
    end
end

function error_analysis(sys::System, e)
    is_time_dependent(sys) || rethrow(e)
    eqs = get_eqs(sys)
    unknowns = get_unknowns(sys)
    t = get_iv(sys)
    println("The system of equations is:")
    println(eqs)
    return if e isa ModelingToolkit.ExtraVariablesSystemException
        rs = [Differential(t)(state) => state for state in unknowns]
        extraunknowns = [state for state in unknowns]
        extraeqs = [eq for eq in eqs]
        numderivs = 0
        for r in rs
            for eq in extraeqs
                if subsmatch(eq.lhs, r) | subsmatch(eq.rhs, r)
                    extraunknowns = vec(setdiff(extraunknowns, [r.second]))
                    extraeqs = vec(setdiff(extraeqs, [eq]))
                    numderivs += 1
                    break
                end
            end
        end
        println()
        println("There are $(length(unknowns)) variables and $(length(eqs)) equations.\n")
        println("There are $numderivs time derivatives.\n")
        println("The variables without time derivatives are:")
        println(extraunknowns)
        println()
        println("The equations without time derivatives are:")
        println(extraeqs)
        rethrow(e)
    else
        rethrow(e)
    end
end
