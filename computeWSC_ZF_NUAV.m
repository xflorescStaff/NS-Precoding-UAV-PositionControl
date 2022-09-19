function [WSC_Val, DeltaComp] = computeWSC_ZF_NUAV(A, E, UAVs, dAB, gammaA, gammaJ, channelParam )
    % Compute WSC considering precoding scheme

    %   A:              Alice's Position        1x3
    %   B:              Bob's Position          1x3
    %   E:              Eve's Positions         nEx3
    %   UAV1:           Position of UAV1        1x3
    %   UAV2:           Position of UAV2        1x3
    %   Rj:             Orbit radius of UAVs    1x1
    %   hj:             UAVs height             1x1
    %   dAB:            Distance from A to B	1x3     Can be actual or estimated
    %   gammaA:         Tx SNR at A             1x1
    %   gammaJ:         Tx SNR at UAVs          1x1
    %   channelParam:   Channel and PL parameters
    
    nUAV = size(UAVs,1);
    N = nUAV/2;
    
    B = A;
    B(1) = B(1) + dAB;
    
    % Channel parameters
    phi         = channelParam(1);
    omega       = channelParam(2);
    alpha       = channelParam(3);
    alpha_AG    = channelParam(4);
    ne_LOS      = channelParam(5);
    ne_NLOS     = channelParam(6);
    Rs          = channelParam(7);
    k           = channelParam(8);      % Number of rays of Nakagami channel (m)                        TO IMPLEMENT
    choice      = channelParam(10);     % Choice of which channel to use(0: Rayleigh, 1: Nakagami-m)    TO IMPLEMENT
    
    % Parameters regarding Eve
    dAE = transpose(sqrt( ( A(:,1) - E(:,1) ).^2 + ( A(:,2) - E(:,2) ).^2 ));
    OmegaAE = dAE.^alpha;

	% UAVs
    dJE = sqrt( ( UAVs(:,1) - E(:,1)' ).^2 + ( UAVs(:,2) - E(:,2)' ).^2  + ( UAVs(:,3) - E(:,3)' ).^2);
    Theta_JE = (180/pi) * asin(UAVs(:,3)./dJE);
    PLOS_JE = 1./(1 + phi * exp( -omega*( Theta_JE - phi ) ) );
    LJE = PLOS_JE.*(abs(dJE).^alpha_AG)*ne_LOS + (1-PLOS_JE).*(abs(dJE).^alpha_AG)*ne_NLOS;
    gJE = 1./LJE;
    hJE = sqrt(gJE);
    
    aNJ = gammaA;
    bNJ = gammaA;
    
    % Parameters regarding Bob
    OmegaAB = dAB.^alpha;
    
    dJB = sqrt( ( UAVs(:,1) - B(1) ).^2 + ( UAVs(:,2) - B(2) ).^2  + ( UAVs(:,3) - B(3) ).^2);
    Theta_JB = (180/pi) * asin(UAVs(:,3)./dJB);
    PLOS_JB = 1./(1 + phi * exp( -omega*( Theta_JB - phi ) ) );
    LJB = PLOS_JB.*(abs(dJB).^alpha_AG)*ne_LOS + (1-PLOS_JB).*(abs(dJB).^alpha_AG)*ne_NLOS;
    gJB = 1./LJB;
    hJB = sqrt(gJB);

    % Precoding interferent channel
    g_INT = 0;
    for i=1:N
       h_INT = hJB(2*i)*hJE(2*i - 1,:) - hJB(2*i - 1)*hJE(2*i,:);
       g_INT = g_INT + abs(h_INT).^2;
    end
    
    % SOP parameters and computation
    aJ = gammaA;
    bJ = gammaA ./ ( 1 + gammaJ*g_INT );
    beta    = ((2.^Rs)-1)./aJ;
    eta     = (2.^Rs)./ ( 1 + gammaJ*g_INT );
    
    switch choice
        case 0
            SOP_J   = 1 - (exp( -(OmegaAB./aJ ).*(2^Rs - 1) ))*( 1 ./ ( (2^Rs)*(OmegaAB./OmegaAE).*( bJ./aJ ) + 1 ) );
            SOP_NJ  = 1 - (exp( -(OmegaAB./aNJ).*(2^Rs - 1) ))*( 1 ./ ( (2^Rs)*(OmegaAB./OmegaAE).*(bNJ./aNJ) + 1 ) );
        case 1
            SOP_J   = SOP_NakagamiM_N(beta, eta, 1./OmegaAB, 1./OmegaAE, k, k ,1);
            SOP_NJ  = SOP_NakagamiM_N((2.^Rs-1)/gammaA, 2.^Rs, 1./OmegaAB, 1./OmegaAE, k, k ,1);
    end
    

    % Area-based metrics
    DeltaComp = (1-SOP_J)./(1-SOP_NJ);

    coverage    = 0;
    efficiency  = 0;
    for i=1:length(DeltaComp(:))
        coverage    = coverage + (DeltaComp(i)>=1);
        efficiency  = efficiency + DeltaComp(i);
    end
    efficiency  = efficiency/length(DeltaComp(:));
    WSC_Val     =   coverage*efficiency;
end