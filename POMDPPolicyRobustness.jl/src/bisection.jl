#Modified Bisection Search Method
function upper_bisection_search(f,xl::Float64,xu::Float64;max_iters=1000,eps=1e-7,verbose=true) #Assumes Monotonicity
    time = 0.0
    n = 0
    delta = Inf
    a = xl
    fa = f(xl)
    vector_flag = false
    if isa(fa,Tuple)
        vector_flag = true
        time += fa[2]
    end
    b = xu
    fb = f(xu)
    if vector_flag 
        time += fb[2]
    end
    if sign(fa[1]) == sign(fb[1])
        verbose && @warn "No root on interval. Using upper boundary value: $fb"
        if !vector_flag 
            return xu
        else
            return xu, 0.0
        end
    end
    verbose && println("N   V    L    U  ")
    while n <= max_iters && delta > eps
        n += 1
        c = (a+b)/2
        fc = f(c)
        if vector_flag
            time += fc[2]
        end
        verbose && println("$n $fc $a $b")
        delta = (b-a)
        if delta < eps
            vector_flag && @info "STORM time for bisection is $time."
            if vector_flag
                return a,time
            else 
                return a
            end
        end
        if sign(fc[1]) == 0 #If on the minimizer, set as LB
            a = c
            fa = fc
        elseif sign(fc[1]) == sign(fa[1]) #If not, move in the bound
            a = c
            fa = fc
        else #if sign(fc) == sign(fb) - Left or skipped over min
            b = c
            fb = fc
        end
    end
end