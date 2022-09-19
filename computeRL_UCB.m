function [WSCEst, WSCN] = computeRL_UCB(WSCEst, hJ, RJ, WSCN, angleUAV, ...
                                            A, E, dAB_R, gammaA, gammaJ, c, channelParam, i, alpha, nUAV, precode, typeA)
    % Compute a RL step

    %   Upper-Confidence-Bound Action Selection
    if i == 1
        ind = randi(length(WSCEst));
    else
        WSCEst_UCB = WSCEst + c*sqrt( log(i)./WSCN );
        maxInds_UCB = find( WSCEst_UCB == max( WSCEst_UCB ) );              %   Store multiple maximum values indeces
        if(length(maxInds_UCB) > 1)                                         %   Check if there are multiple maximum values
            ind = maxInds_UCB(randi(length(maxInds_UCB)));                 	%   Choose a random greedy action
        else
            ind = maxInds_UCB;                                              %   Choose (single) greedy action with 1-epsilon probability
        end
    end
    
    % Update virtual UAVs positions
    indAng = ind;
    Angle   = angleUAV(indAng);
    
    UAVs = setNewPos_N(nUAV, Angle, hJ, RJ, typeA);
    
    %   Compute Reward (WSC) of action ind
    if precode==1
        WSC = computeWSC_ZF_NUAV(A, E, UAVs, dAB_R, gammaA, gammaJ, channelParam )/length(E(:));
    elseif precode==2
        WSC = computeWSC_MRT_NUAV(A, E, UAVs, dAB_R, gammaA, gammaJ, channelParam )/length(E(:));
	elseif precode==3
        WSC = computeWSC_NOP_NUAV(A, E, UAVs, dAB_R, gammaA, gammaJ, channelParam )/length(E(:));
	elseif precode==4
        WSC = computeWSC_NSMRT_NUAV(A, E, UAVs, dAB_R, gammaA, gammaJ, channelParam )/length(E(:));
    end
    
    %   Action-value updates
    WSCN(ind) = WSCN(ind)+1;                                                %   Update the ocurrences
    if alpha >0
        WSCEst(ind) = WSCEst(ind) + alpha*(WSC-WSCEst(ind));                %   Action value incremental update with fixed step size
    else
        WSCEst(ind) = WSCEst(ind) + (1/WSCN(ind))*(WSC-WSCEst(ind));        %   Action value incremental update
    end

end