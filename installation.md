# installation

#julia



# install julia
`sudo mkdir /opt/julia-1.0.2`
`wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.2-linux-x86_64.tar.gz`
`echo "e0e93949753cc4ac46d5f27d7ae213488b3fef5f8e766794df0058e1b3d2f142 *julia-1.0.2-linux-x86_64.tar.gz" | sha256sum -c`-
`sudo tar xzf julia-1.0.2-linux-x86_64.tar.gz -C /opt/julia-1.0.2 --strip-components=1`
`rm /tmp/julia-1.0.2-linux-x86_64.tar.gz`
`ln -fs /opt/julia-1.0.2/bin/julia /usr/local/bin/julialang`

# install julia packages 
```
julia -e 'import Pkg; Pkg.update()' && \
julia -e 'import Pkg; Pkg.add("LightGraphs")' && \
julia -e 'import Pkg; Pkg.add("Optim")' && \
julia -e 'import Pkg; Pkg.add("BinDeps")' && \
julia -e 'import Pkg; Pkg.add("NPZ")' $$ \
julia -e 'import Pkg; Pkg.add("JuMP")' $$ \	
julia -e 'import Pkg; Pkg.add("Ipopt")' $$ \	
julia -e 'import Pkg; Pkg.add("PyCall") $$ \
import Pkg; Pkg.add("Gurobi")'
```

```
wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.2-linux-x86_64.tar.gz
echo "e0e93949753cc4ac46d5f27d7ae213488b3fef5f8e766794df0058e1b3d2f142 *julia-1.0.2-linux-x86_64.tar.gz" | sha256sum -c
sudo tar xzf julia-1.0.2-linux-x86_64.tar.gz -C /opt/julia-1.0.2 --strip-components=1
rm /tmp/julia-1.0.2-linux-x86_64.tar.gz
sudo ln -fs /opt/julia-1.0.2/bin/julia /usr/local/bin/julialang
```