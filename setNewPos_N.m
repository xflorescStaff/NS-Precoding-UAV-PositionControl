function UAVs = setNewPos_N(nUAV, Ang, h, Rj, typeA)
    % Obtain the positioning of nUAV UAVs
    
    UAVs = zeros(nUAV, 3);
    switch typeA
        case 1
            % Symmetric around A and B
            phi = pi - ((nUAV-1)/2)*Ang;                    % Angle of first UAV that preserves symmetry
            for iU = 1:nUAV
                UAVs(iU,1) = Rj*cos(phi + (iU-1)*Ang);
                UAVs(iU,2) = Rj*sin(phi + (iU-1)*Ang);
                UAVs(iU,3) = h;
            end
        case 2
            % Opening towards one side of A-B
            phi = pi - ((nUAV-1))*Ang;
            for iU = 1:nUAV
                UAVs(iU,1) = Rj*cos(phi + (nUAV-iU)*Ang);
                UAVs(iU,2) = Rj*sin(phi + (nUAV-iU)*Ang);
                UAVs(iU,3) = h;
            end
    end
end