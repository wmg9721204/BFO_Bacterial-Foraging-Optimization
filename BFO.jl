using LinearAlgebra ## "norm"
using StatsBase ## "sample"
"""
**BFO** is a function that computes the **classical (non-adaptive) BFO (Bacterial Foraging Optimization)**<br><br>
**Inputs**:
1. **J** = a function with domain R^n
2. **Range** = [-10,10], exploration range: Range^n <br><br>

3. **n** = 2, dimension of the input of J
4. **S** = 10, number of bacteria
5. **Sr** = 4,  number of bacteria removed in reproductive step
6. **Nc** = 20, number of chemotactic steps
7. **Ns** = 5, number of swim steps
8. **Nre** = 5, number of reproductive steps
9. **Ned** = 2, elimination and dispersal steps
10. **Ped** = 0.3, probability of elimination
11. **Ci** = (Range[2]-Range[1])/S, the run-length unit
**Output**: a dictionary that stores
1. the minimum value of J
2. the point achieving this minimu value
3. the path of each bacterium (for plotting illustration)
"""
function BFO(J, Range; n = 2::Int, S = 10::Int, Sr = 4::Int, Nc = 20::Int, Ns = 5::Int, 
        Nre = 5::Int, Ned = 2::Int, Ped = 0.3::Float64, Ci = ((Range[2]-Range[1])/S)::Float64)
    
    ## randomly generate S bacteria in Range
    Rand = rand(n,S) ## generate a random template
    B_loc = zeros(n,S)  ## B_loc = Bacteria locations
    for d = 1:n
        B_loc[d,:] = (Range[d,2]-Range[d,1])*Rand[d,:].+Range[d,1]
    end
    B_loc
    
    ## a dictionary recording the path of bacterium i
    Path_Dict = Dict(i=>[zeros(n,0) B_loc[:,i]] for i=1:S)
    for l = 1:Ned ## index of elimination-dispersal steps
        for k = 1:Nre ## index of reproductive steps
            for j = 1:Nc ## index of chemotactic steps
                ## Chemotactic Step
                println("Chemo-step $j")
                for i = 1:S ## index of bacterium
                    ## Tumble/Swim
                    Path_i = copy(B_loc[:,i]) ## record the path of bacterium i
                    m = 0 ## counter for swimming
                    while m<Ns
                        delta_i = randn(n) ## random "tumble" direction (uniform on (n-1)-sphere)
                        B_i_loc_new = B_loc[:,i] + Ci*delta_i/norm(delta_i) ## new location
                        ## A while loop to guarantee the new bacterial position is in Range
                        while sum((Range[:,1].<=B_i_loc_new).&(B_i_loc_new.<=Range[:,2]))!=n
                            delta_i = randn(n) ## random "tumble" direction (uniform on (n-1)-sphere)
                            B_i_loc_new = B_loc[:,i] + Ci*delta_i/norm(delta_i) ## new location
                        end
                        J_last = J(B_loc[:,i]) ## last fitness value
                        J_new = J(B_i_loc_new) ## new fitness value
                        if J_new<J_last ## swim
                            m = m+1
                            J_last = copy(J_new)
                            B_loc[:,i] = B_i_loc_new
                            ## update the path of bacterium i
                            Path_i = [Path_i B_i_loc_new]
                        else
                            m = Ns ## don't swim 
                        end
                    end
                    ## update the path of bacterium i
                    if Path_Dict[i][:,end]!=B_loc[:,i]
                        Path_Dict[i] = [Path_Dict[i] B_loc[:,i]]
                    end
                end
            end
            
            if k<Nre
                ## Reproductive Step
                println("Repro-step $k")
                ## (!!!) I define the health of a bacterium as the J value of its current location
                Health = [J(B_loc[:,i]) for i=1:S]
                Health_sort = sortslices([Health collect(1:S)], dims = 1) ## sort the bacteria according to health
                B_survive = Array{Int,1}(Health_sort[1:S-Sr,2]) ## ## pick out the most healthy (S-Sr) bacteria
                B_rep = sort([B_survive;B_survive[1:Sr]]) ## reproduce the most healthy Sr bacteria
                B_loc = B_loc[:,B_rep]
                
                ## update the path dictionary
                for i = 1:S
                    if Path_Dict[i][:,end]!=B_loc[:,i]
                        Path_Dict[i] = [Path_Dict[i] B_loc[:,i]]
                    end
                end
            end
        end
        ## Elimination-Dispersal Step
        println("ED-step $l")
        if l<Ned
            KillAlive = [sample([false, true], aweights([Ped,1-Ped])) for i=1:S] # true = alive, false = kill
            N_kill = S - sum(KillAlive) ## number of kiilled bacteria
            ## randomly generate a bacterium for each killed bacterium
            Alive = findall(KillAlive)
            Kill = setdiff(collect(1:S),Alive)
            ## Rebuild the killed bacteria in random locations in Range
            ED = [Alive'; B_loc[:,Alive]]
            for dead in Kill
                Revive_dead = [dead;[(Range[d,2]-Range[d,1])*rand(1)[1]+Range[d,1] for d=1:n]]
                ED = [ED Revive_dead]
            end
            B_loc = sortslices(ED, dims = 2)[2:n+1,:]
            ## update the path dictionary
            for i = 1:S
                if (Path_Dict[i][:,end])!=B_loc[:,i]
                    Path_Dict[i] = [Path_Dict[i] B_loc[:,i]]
                end
            end
        end
    end
    B_best = Int(sortslices([[J(B_loc[:,i]) for i=1:S] collect(1:S)], dims = 1)[1,2]) ## best bacterium
    X_best = B_loc[:,B_best] ## best location for minimizing J
    J_best = J(X_best) ## best (minimum) J value
    return Dict("Minimum"=>J_best, "Minimum Point"=>X_best, "Path_Dict"=>Path_Dict)
end