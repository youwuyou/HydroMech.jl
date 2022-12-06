using HydroMech
using BenchmarkTools

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)


function bc1!(ητ)
    nx, ny = size(ητ)
    @parallel (1:nx) free_slip_y!(ητ)
    @parallel (1:ny) free_slip_x!(ητ)
end

function bc2!(ητ)
    @parallel (1:size(ητ,2)) free_slip_x!(ητ)
    @parallel (1:size(ητ,1)) free_slip_y!(ητ)
end

function bc3!(ητ, nx, ny)
    @parallel (1:ny) free_slip_x!(ητ)
    @parallel (1:nx) free_slip_y!(ητ)
end


function main()

    nx, ny = 1000, 1000
    ητ     = rand(Float64, (nx,ny))

    # benchmark the three routines that gets called
    @btime bc1!($ητ)
    @btime bc2!($ητ)
    @btime bc3!($ητ, $nx, $ny)
end



main()