using LinearAlgebra ## "norm"
using StatsBase ## for function "sample"
## This function computes the classical (non-adaptive) BFO (Bacterial Foraging Optimization)
## Inputs:
## J = a function with domain R^n
## n = 2, dimension of the input of J
## Range = [-10,10], exploration range: Range^n
## S = 10, number of bacteria
## Sr = number of bacteria removed in reproductive step
## Nc = number of chemotactic steps
## Ns = number of swim steps
## Nre = number of reproductive steps
## Ned = elimination and dispersal steps
## Ped = probability of elimination
## Ci = (Range[2]-Range[1])/S; ## run-length unit
## Output: a dictionary that stores
## (1) the minimum value of J
## (2) the point achieving this minimu value
## (3) the path of each bacterium (for plotting illustration)
function BFO(J, Range, n = 2::Int, S = 10::Int, Sr = 4::Int, Nc = 20::Int, Ns = 5::Int, 
        Nre = 50::Int, Ned = 10::Int, Ped = 0.3::Float64, Ci = ((Range[2]-Range[1])/S)::Float64)
    ## randomly generate S bacteria in Range^n
    B_loc = (Range[2]-Range[1])*rand(n,S).+Range[1] ## B_loc = Bacteria locations
    ## a dictionary recording the path of bacterium i
    Path_Dict = Dict(i=>zeros(n,0) for i=1:S)
    for l = 1:Ned ## index of elimination-dispersal steps
        for k = 1:Nre ## index of reproductive steps
            for j = 1:Nc ## index of chemotactic steps
                ## Chemotactic Step
                for i = 1:S ## index of bacterium
                    ## Tumble/Swim
                    Path_i = copy(B_loc[:,i]) ## record the path of bacterium i
                    m = 0 ## counter for swimming
                    delta_i = randn(n) ## random "tumble" direction (uniform on (n-1)-sphere)
                    while m<Ns
                        J_last = J(B_loc[:,i]) ## last fitness value
                        B_i_loc_new = B_loc[:,i] + Ci*delta_i/norm(delta_i) ## new location
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
                    Path_Dict[i] = [Path_Dict[i] Path_i] ## update the path of bacterium i
                end
            end
            if k<Nre
                ## Reproductive Step
                ## (!!!) I define the health of a bacterium as the J value of its current location
                Health = [J(B_loc[:,i]) for i=1:S]
                Health_sort = sortslices([Health collect(1:S)], dims = 1) ## sort the bacteria according to health
                B_survive = Array{Int,1}(Health_sort[1:S-Sr,2]) ## ## pick out the most healthy (S-Sr) bacteria
                B_rep = sort([B_survive;B_survive[1:Sr]]) ## reproduce the most healthy Sr bacteria
                B_loc = B_loc[:,B_rep]
                
                ## update the path dictionary
                for i = 1:S
                    Path_Dict[i] = [Path_Dict[i] B_loc[:,i]]
                end
            end
        end
        ## Elimination-Dispersal Step
        if l<Ned
            KillAlive = [sample([false, true], aweights([Ped,1-Ped])) for i=1:S] # true = alive, false = kill
            N_kill = S - sum(KillAlive) ## number of kiilled bacteria
            ## randomly generate a bacterium for each killed bacterium
            Alive = findall(KillAlive)
            Kill = setdiff(collect(1:S),Alive)
            B_loc = sortslices([[Alive'; B_loc[:,Alive]] [Kill';(Range[2]-Range[1])*rand(n,length(Kill)).+Range[1]]], dims = 2)[2:n+1,:]
            ## update the path dictionary
            for i = 1:S
                Path_Dict[i] = [Path_Dict[i] B_loc[:,i]]
            end
        end
    end
    B_best = Int(sortslices([[J(B_loc[:,i]) for i=1:S] collect(1:S)], dims = 1)[1,2]) ## best bacterium
    X_best = B_loc[:,B_best] ## best location for minimizing J
    J_best = J(X_best) ## best (minimum) J value
    return Dict("Minimum"=>J_best, "Minimum Point"=>X_best, "Path_Dict"=>Path_Dict)
end