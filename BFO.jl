using LinearAlgebra ## "norm"
using StatsBase ## "sample"

## m: number of random points to generate
## Range: dim x 2 matrix; the d-th row is the range of the d-th variable
function RandPts(Range::Array{Float64,2}, m::Int)
    dim = size(Range,1)
    randPts = rand(dim,m)
    for d = 1:dim
        Original = randPts[d,:]
        randPts[d,:] = (Range[d,2]-Range[d,1])*Original.+Range[d,1]
    end
    return randPts
end

"""
**RandUnit** is a function generating m random unit vectors in dimension d. <br>
**Inputs**: <br>
1. d: dimension <br>
2. m: number of vectors to generate <br>
**Output**: <br>
an d x m matrix whose columns are the desired vectors. <br>
"""
function RandUnit(d,m)
    Rand = randn(d,m)
    for k = 1:m
        Rand[:,k]./=norm(Rand[:,k])
    end
    return Rand
end

## Some parts have been vectorized to make implementation faster. 
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
function BFO(J, Range, n = 2::Int, S = 10::Int, Sr = 4::Int, Nc = 20::Int, Ns = 5::Int, 
        Nre = 5::Int, Ned = 2::Int, Ped = 0.3::Float64, Ci = ((Range[2]-Range[1])/S)::Float64)
    
    ## randomly generate S bacteria in Range
    B_loc = RandPts(Range,S)
    ## a dictionary recording the path of bacterium i
    Path_Dict = Dict(i=>[zeros(n,0) B_loc[:,i]] for i=1:S)
    for l = 1:Ned ## index of elimination-dispersal steps
        for k = 1:Nre ## index of reproductive steps
            for j = 1:Nc ## index of chemotactic steps
                ## Chemotactic Step
                println("Chemo-step $j")
                ToSwim = collect(1:S) ## monitor the swimming bacteria
                Tumble = RandUnit(n,S)
                m = 0 ## index of swimming
                while (ToSwim!=Array{Int64,1}())&(m<Ns) ## tumble/swim
                    m = m + 1
                    J_old = [J(B_loc[:,i]) for i in ToSwim]
    
                    B_loc_new = copy(B_loc)
                    B_loc_new[:,ToSwim] = B_loc[:,ToSwim].+Ci*Tumble[:,ToSwim]
                    ## Mirror back the out-of-range bacteria
                    for d = 1:n
                        ## mirror back bacteria tumbling out from below
                        B_loc_new[d,ToSwim] = (abs.(B_loc_new[d,ToSwim].-Range[d,1])).+(Range[d,1]) 
                        ## mirror back bacteria tumbling out from above
                        B_loc_new[d,ToSwim] = Range[d,2].-(abs.(Range[d,2].-B_loc_new[d,ToSwim]))
                    end
                    B_loc_new
    
                    ## Evaluate J at new locations
                    J_new = [J(B_loc_new[:,i]) for i in ToSwim]
    
                    ## Find out improved bacteria and update the swimming ones
                    ToSwim = ToSwim[findall(J_new.<J_old)]
    
                    for i in ToSwim
                        Path_Dict[i] = [Path_Dict[i] B_loc_new[:,i]]
                    end
                    B_loc[:,ToSwim] = B_loc_new[:,ToSwim]
                end ## end of tumble/swim
            end ## end of chemotactic steps
            
            if k<Nre
                ## Reproductive Step
                println("Repro-step $k")
                ## (!!!) I define the health of a bacterium as the J value of its current location
                Health = [J(B_loc[:,i]) for i=1:S]
                Health_sort = sortslices([Health collect(1:S)], dims = 1) ## sort the bacteria according to health
                B_survive = Array{Int,1}(Health_sort[1:S-Sr,2]) ## ## pick out the most healthy (S-Sr) bacteria
                B_die = setdiff(collect(1:S), B_survive)
                B_rep = sort([B_survive;B_survive[1:Sr]]) ## reproduce the most healthy Sr bacteria
                B_loc[:,B_die] = B_loc[:,B_survive[1:Sr]]
                
                ## update the path dictionary
                for i in B_die
                    Path_Dict[i] = [Path_Dict[i] B_loc[:,i]]
                end
            end
        end
        ## Elimination-Dispersal Step
        println("ED-step $l")
        if l<Ned
            Alive = [sample([false, true], aweights([Ped,1-Ped])) for i=1:S] # true = alive, false = kill
            Killed = setdiff(collect(1:S),Alive) ## kiilled bacteria
            ## randomly generate a bacterium for each killed bacterium
            Alive = findall(Alive)
            
            ## Rebuild the killed bacteria in random locations in Range
            B_loc[:,Killed] = RandPts(Range,length(Killed))
            ## update the path dictionary
            for i in Killed
                Path_Dict[i] = [Path_Dict[i] B_loc[:,i]]
            end
        end
    end
    B_best = Int(sortslices([[J(B_loc[:,i]) for i=1:S] collect(1:S)], dims = 1)[1,2]) ## best bacterium
    X_best = B_loc[:,B_best] ## best location for minimizing J
    J_best = J(X_best) ## best (minimum) J value
    return Dict("Minimum"=>J_best, "Minimum Point"=>X_best, "Path_Dict"=>Path_Dict)
end