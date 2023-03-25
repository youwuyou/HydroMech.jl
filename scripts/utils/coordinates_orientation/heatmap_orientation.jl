using Plots

const HEATMAP_DEFAULT_ORIENTATION = false
const WITH_X_Y  = true

mymatrix = zeros(Int, (7,5))
mymatrix[1,1] = 1


@show size(mymatrix)
@show mymatrix


if HEATMAP_DEFAULT_ORIENTATION && WITH_X_Y == false
    heatmap(mymatrix', color = :greys)
    # savefig("heatmap_debug_default.png")
    savefig("heatmap_debug_transpose.png")

else
    heatmap(mymatrix, color = :greys, yflip=true)
    savefig("heatmap_debug_with_y_flip.png")

end



if WITH_X_Y

        # Xv          = (-dx/2):dx:(lx+dx/2)

        # X           = 0:dx:lx 
        # Y           = 0:dy:ly

        # Yv          = (-dy/2):dy:(ly+dy/2)

        # % STAGGERED GRID
        # x           = 0:dx:xsize+dx;            
        # y           = 0:dy:ysize+dy;            
        # xvx         = 0:dx:xsize+dx;
        # yvx         = -dy/2:dy:ysize+dy/2;
        # xvy         = -dx/2:dx:xsize+dx/2;   
        # yvy         = 0:dy:ysize+dy;        
        # xpr         = -dx/2:dx:xsize+dx/2;
        # ypr         = -dy/2:dy:ysize+dy/2;




    # X, Y, Yv = 0:dx:lx, 0:dy:ly, (-dy/2):dy:(ly+dy/2)


    X    = 0:1:4
    Y    = 0:2:9

    heatmap(X,Y,mymatrix, color= :greys,xlims=(X[1],X[end]), ylims=(Y[1],Y[end]),  yflip=true)
    savefig("heatmap_debug_with_xandy.png")


end



# julia> mymatrix
# 5x5 Matrix{Int64}
#  1  0  0  0  0
#  0  0  0  0  0
#  0  0  0  0  0
#  0  0  0  0  0
#  0  0  0  0  0
