function [WSC_Max, Ang_Max, DeltaComp_Max] = optimalWSC_ZF(A, E, Rj, hj, dAB, gammaA, gammaJ, channelParam, thetaV, nUAV, typeA )
    % Compute optimal WSC through exhaustive search (precoding)

    WSC_Max = 0;
    Ang_Max = 0;
    DeltaComp_Max = 0;
    for ang=thetaV
        UAVs = setNewPos_N(nUAV, ang, hj, Rj, typeA);
        [WSC_Dum, DeltaComp] = computeWSC_ZF_NUAV(A, E, UAVs, dAB, gammaA, gammaJ, channelParam );
        if WSC_Dum > WSC_Max
            WSC_Max = WSC_Dum;
            Ang_Max = ang;
            DeltaComp_Max = DeltaComp;
        end 
    end 
end