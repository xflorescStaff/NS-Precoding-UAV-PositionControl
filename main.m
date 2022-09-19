
%% gammaJ - nUAV
clc, clear, close all
gammaJ_V    = [10, 50, 100, 500, 1000];
nUAV_V      = [2, 4, 6];
verStr = 'PreTy-v00R.mat';

vecMain = gammaJ_V;
vecVals = nUAV_V;
nMain = length(vecMain);
nVals = length(vecVals);
strMain = 'gammaJ';
strVals = 'nUAV';

texMain = '$\gamma_{\mathrm{J}}$';
texVals = '$nUAV$';

nMC = 100;
typeA = 2 ;         % 1: symmetric angles, 2: semicircle angles 
% ----------------------------------------------------------------------

% Environment parameters (Urban)
phi = 9.61;         % Environmental constant
omega = 0.16 ;      % Environmental constant
alpha = 0.3;        % Ground Path Loss exponent
alpha_AG = 0.3;     % Air-to-Ground Path Loss Exponent
ne_LOS = 1.0;       % Air-to-Ground LOS attenuation
ne_NLOS = 20;       % Air-to-Ground NLOS attenuation

% Channel parameters
m = 3;              % Number of parallel channels
sigma = 1/sqrt(2);  % Noise std dev of each component (real and imag) of every parallel channel
choice = 1;         % 0: Rayleigh channel, 1: Nakagami-m channel
Rs = 1;
channelParam =  [   phi,...
                    omega,...
                    alpha,...
                    alpha_AG,...
                    ne_LOS,...
                    ne_NLOS,...
                    Rs,...
                    m,...
                    sigma,...
                    choice];

% Positions ***************************************************************
    % Alice
A = [0,0,0];        %   Position of Alice (zero point)
gammaA = 100;       %   Alice Tx SNR

    % Bob
sigAB = 1;          %   Unreliability of B's position

    % Eve
nR = 50;            %   Number of radial points
nTheta = 180;       %   Number of angular points
nE = nR*nTheta;     %   Number of Eves

rLow = 0.1;         %   Lowest radius of Eve
rHigh = 50;         %   Highest radius of Eve
thetaLow = 0;       %   Lowest angle of Eve
thetaHigh = 2*pi;     %   Highest radius of Eve

rangeR = linspace(rLow,rHigh,nR);                       % Points in Radial dimension
rangeTheta = linspace(thetaLow,thetaHigh,nTheta);       % Points in Angular dimension

thetat  =  repmat(rangeTheta,1,nR);
rt  = (repmat(rangeR',1,nTheta)).';

E = [rt(:).*cos(thetat(:)), rt(:).*sin(thetat(:)) , zeros(nR*nTheta,1)];             % Eves' position (rectangle coordinates)

% -------------------------------------------------------------------------
    % UAV

nAng = 10;                                          %   Angle discretization level (opening angle)  -> Number of Angle Actions

% UAV = 2;                                            %   Number of simultaneous UAVs
% angleUAV    = linspace(0,2*pi/(nUAV-1),nAng);       %   Possible angle actions (opening angles)

hj = 30;                            % Fixed
Rj = 40;                            % Fixed

% -------------------------------------------------------------------------

% *************************************************************************

%   k-Armed Bandits
nLoops = 20;               % Number of loops for action choosing
initWSC = 0;                % Optimistic initial action values
c = 0.3;                    % Exploration parameter for UCB
alpha = 0.1;                % Step size (0: uniform average)

%   Performance Variables
dt = 0;
alphat = 0.2;

WSC_RL      = zeros(nMC,nLoops,nMain,nVals);
WSC_NOP     = zeros(nMC,nLoops,nMain,nVals);
WSC_GD      = zeros(nMC,nLoops,nMain,nVals);
WSC_Max_ZF  = zeros(nMC,nLoops,nMain,nVals);
WSC_Max_NOP = zeros(nMC,nLoops,nMain,nVals);

Ang_RL_V        = zeros(nMC,nLoops,nMain,nVals);
Ang_NOP_V       = zeros(nMC,nLoops,nMain,nVals);
Ang_Step_V      = zeros(nMC,nLoops,nMain,nVals);
Ang_Max_ZF_V    = zeros(nMC,nLoops,nMain,nVals);
Ang_Max_NOP_V   = zeros(nMC,nLoops,nMain,nVals);

dists_AB = zeros(nMC,nLoops,nMain,nVals);

% /*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*
for iMC =1:nMC
    
    % Bob's movement
    xo = 0;                     
    xf = 0;
    while abs(xo-xf)<=nLoops*1e-2   % Do not allow movements that are too short (such that each step is more than 0.01)
        xo = rHigh*rand();          % Initial position
        xf = rHigh*rand();          % Final position
    end
    dAB_R_V = linspace(xo,xf,nLoops);       % Total trajectory for all Loops
    
    for iVm = 1:nMain
        
        tic
        for iVa = 1:nVals
            nUAV = vecVals(iVa);
            gammaJ = vecMain(iVm)/nUAV;
            if typeA==1
                angleUAV    = linspace(0,2*pi/(nUAV-1),nAng);       %   Possible angle actions (opening angles)
            elseif typeA==2
                angleUAV    = linspace(0,pi/(nUAV-1),nAng);
            end
                        
            % /*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*
                % Initialization for RL process
            WSCEst_Angle    = initWSC*ones(1,nAng);         %   Action value estimation vector for angle actions
            WSCN_Angle      = zeros(1,nAng);                %   Vector to store angle action ocurrences

            WSCEst_Ang_NOP    = initWSC*ones(1,nAng);         %   Action value estimation vector for angle actions
            WSCN_Ang_NOP      = zeros(1,nAng);                %   Vector to store angle action ocurrences

            WSC_Ang         = angleUAV(fix(length(angleUAV)/2));       %   Angle initialization
            WSC_UAV = 0;                                    %   WSC initial value

            % Initialization for GD process
            Ang_Step = WSC_Ang;                             %   The same as the RL approach, to better compare the two approaches

            for i=1:nLoops
                dAB_R = dAB_R_V(i);
                
                % Exhaustive Search results
                [WSC_Max_Val, Ang_Max_Val, ~]       = optimalWSC_ZF(A, E, Rj, hj, dAB_R, gammaA, gammaJ, channelParam, angleUAV, nUAV, typeA );
                [WSC_Max_ValNOP, Ang_Max_ValNOP, ~] = optimalWSC_NOP(A, E, Rj, hj, dAB_R, gammaA, gammaJ, channelParam, angleUAV, nUAV, typeA );

                WSC_Max_ZF(iMC,i,iVm, iVa)       = WSC_Max_Val;
                Ang_Max_ZF_V(iMC,i,iVm, iVa)     = Ang_Max_Val;

                WSC_Max_NOP(iMC,i,iVm, iVa)      = WSC_Max_ValNOP;
                Ang_Max_NOP_V(iMC,i,iVm, iVa)    = Ang_Max_ValNOP;
                
                % dAB estimation and parameter computation
                dAB = normrnd(dAB_R,sigAB);                             %   CSI estimate

                %   RL iteration - ZF
                [WSCEst_Angle, WSCN_Angle]      = computeRL_UCB(WSCEst_Angle, hj, Rj, WSCN_Angle, angleUAV, ...
                                                    A, E, dAB, gammaA, gammaJ, c, channelParam, i, alpha, nUAV,1, typeA );

                %   RL iteration - NOP
                [WSCEst_Ang_NOP, WSCN_Ang_NOP]  = computeRL_UCB(WSCEst_Ang_NOP, hj, Rj, WSCN_Ang_NOP, angleUAV, ...
                                                    A, E, dAB, gammaA, gammaJ, c, channelParam, i, alpha, nUAV,3, typeA);
                
                %   GD iteration
                Ang_Step                        = computeGD(A, E, Rj, hj, dAB, gammaA, gammaJ, channelParam, alpha, Ang_Step, i, nUAV, typeA );

                %   True (Greedy) WSC calculations

                % ZF
                [~, Ang_RL_Ind]                 = max(WSCEst_Angle);
                Ang_RL                          = angleUAV(Ang_RL_Ind);
                UAVs                            = setNewPos_N(nUAV, Ang_RL, hj, Rj, typeA);
                WSC_RL(iMC,i,iVm, iVa)          = computeWSC_ZF_NUAV(A, E, UAVs, dAB_R, gammaA, gammaJ, channelParam );

                % NOP
                [~, Ang_NOP_Ind]                = max(WSCEst_Ang_NOP);
                Ang_NOP                         = angleUAV(Ang_NOP_Ind);
                UAVs                            = setNewPos_N(nUAV, Ang_NOP, hj, Rj, typeA);
                WSC_NOP(iMC,i,iVm, iVa)         = computeWSC_NOP_NUAV(A, E, UAVs, dAB_R, gammaA, gammaJ, channelParam );

                % GD
                UAVs                            = setNewPos_N(nUAV, Ang_Step, hj, Rj, typeA);
                WSC_GD(iMC,i,iVm, iVa)          = computeWSC_ZF_NUAV(A, E, UAVs, dAB_R, gammaA, gammaJ, channelParam );

                Ang_RL_V(iMC,i,iVm, iVa)        = Ang_RL;
                Ang_NOP_V(iMC,i,iVm, iVa)       = Ang_NOP;
                Ang_Step_V(iMC,i,iVm, iVa)      = Ang_Step;

                dists_AB(iMC,i,iVm, iVa)        = dAB_R;
            end
            
            % /*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*
        end
        % Time
        t1 = toc;
        if iMC*iVm == 1
            dt = t1;
        else
            dt = dt + alphat*(t1 - dt);
        end

        TT = dt*nMC*nMain;
        TTF = TT - ( (iMC-1)*nMain + iVm )*dt;

        TTF_S = rem(TTF,60);
        TTF_M = rem(fix(TTF/60),60);
        TTF_H = fix(fix(TTF/60)/60);

        fprintf('MC Loop: %i / %i\t\t %s: %.3f \t\t %s: %.3f \t\t  Time per Loop: %.2f s\t\t TTF: %i H %i M %.1f S \n',iMC,nMC,strMain,vecMain(iVm),strVals,vecVals(iVa),t1,TTF_H,TTF_M,TTF_S);
    end
    
end
% save(['TVT_Plan-',strMain, '-',strVals, '-AngT-', num2str(typeA), '-', verStr ])

%% h - gammaJ
clc, clear, close all
hj_V        = 10:10:100;
gammaJ_V    = [10, 100, 1000];
verStr = 'PreTy-v00R.mat';

vecMain = hj_V;
vecVals = gammaJ_V;
nMain = length(vecMain);
nVals = length(vecVals);
strMain = 'hj';
strVals = 'gammaJ';

texMain = '$h_{\mathrm{J}}$';
texVals = '$\gamma_{\mathrm{J}}$';

nMC = 100;
typeA = 2 ;         % 1: symmetric angles, 2: semicircle angles 
% ----------------------------------------------------------------------

% Environment parameters (Urban)
phi = 9.61;         % Environmental constant
omega = 0.16 ;      % Environmental constant
alpha = 0.3;        % Ground Path Loss exponent
alpha_AG = 0.3;     % Air-to-Ground Path Loss Exponent
ne_LOS = 1.0;       % Air-to-Ground LOS attenuation
ne_NLOS = 20;       % Air-to-Ground NLOS attenuation

% Channel parameters
m = 3;              % Number of parallel channels
sigma = 1/sqrt(2);  % Noise std dev of each component (real and imag) of every parallel channel
choice = 1;         % 0: Rayleigh channel, 1: Nakagami-m channel
Rs = 1;
channelParam =  [   phi,...
                    omega,...
                    alpha,...
                    alpha_AG,...
                    ne_LOS,...
                    ne_NLOS,...
                    Rs,...
                    m,...
                    sigma,...
                    choice];

% Positions ***************************************************************
    % Alice
A = [0,0,0];        %   Position of Alice (zero point)
gammaA = 100;       %   Alice Tx SNR

    % Bob
sigAB = 1;          %   Unreliability of B's position

    % Eve
nR = 50;            %   Number of radial points
nTheta = 180;       %   Number of angular points
nE = nR*nTheta;     %   Number of Eves

rLow = 0.1;         %   Lowest radius of Eve
rHigh = 50;         %   Highest radius of Eve
thetaLow = 0;       %   Lowest angle of Eve
thetaHigh = 2*pi;     %   Highest radius of Eve

rangeR = linspace(rLow,rHigh,nR);                       % Points in Radial dimension
rangeTheta = linspace(thetaLow,thetaHigh,nTheta);       % Points in Angular dimension

thetat  =  repmat(rangeTheta,1,nR);
rt  = (repmat(rangeR',1,nTheta)).';

E = [rt(:).*cos(thetat(:)), rt(:).*sin(thetat(:)) , zeros(nR*nTheta,1)];             % Eves' position (rectangle coordinates)

% -------------------------------------------------------------------------
    % UAV

nAng = 10;                                          %   Angle discretization level (opening angle)  -> Number of Angle Actions
nUAV = 2;                                           %   Number of simultaneous UAVs

% angleUAV    = linspace(0,2*pi/(nUAV-1),nAng);       %   Possible angle actions (opening angles)

% hj = 70;                            % Fixed
Rj = 30;                            % Fixed

% -------------------------------------------------------------------------

% *************************************************************************

%   k-Armed Bandits
nLoops = 20;               % Number of loops for action choosing
initWSC = 0;                % Optimistic initial action values
c = 0.3;                    % Exploration parameter for UCB
alpha = 0.1;                % Step size (0: uniform average)

%   Performance Variables
dt = 0;
alphat = 0.2;

WSC_RL      = zeros(nMC,nLoops,nMain,nVals);
WSC_NOP     = zeros(nMC,nLoops,nMain,nVals);
WSC_GD      = zeros(nMC,nLoops,nMain,nVals);
WSC_Max_ZF  = zeros(nMC,nLoops,nMain,nVals);
WSC_Max_NOP = zeros(nMC,nLoops,nMain,nVals);

Ang_RL_V        = zeros(nMC,nLoops,nMain,nVals);
Ang_NOP_V       = zeros(nMC,nLoops,nMain,nVals);
Ang_Step_V      = zeros(nMC,nLoops,nMain,nVals);
Ang_Max_ZF_V    = zeros(nMC,nLoops,nMain,nVals);
Ang_Max_NOP_V   = zeros(nMC,nLoops,nMain,nVals);

dists_AB = zeros(nMC,nLoops,nMain,nVals);

% /*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*

for iMC =1:nMC
    
    % Bob's movement
    xo = 0;                     
    xf = 0;
    while abs(xo-xf)<=nLoops*1e-2   % Do not allow movements that are too short (such that each step is more than 0.01)
        xo = rHigh*rand();          % Initial position
        xf = rHigh*rand();          % Final position
    end
    dAB_R_V = linspace(xo,xf,nLoops);       % Total trajectory for all Loops
    
    for iVm = 1:nMain
        hj = vecMain(iVm);
        tic
        for iVa = 1:nVals
            gammaJ = vecVals(iVa);
            if typeA==1
                angleUAV    = linspace(0,2*pi/(nUAV-1),nAng);       %   Possible angle actions (opening angles)
            elseif typeA==2
                angleUAV    = linspace(0,pi/(nUAV-1),nAng);
            end
                        
            % /*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*
                % Initialization for RL process
            WSCEst_Angle    = initWSC*ones(1,nAng);         %   Action value estimation vector for angle actions
            WSCN_Angle      = zeros(1,nAng);                %   Vector to store angle action ocurrences

            WSCEst_Ang_NOP    = initWSC*ones(1,nAng);         %   Action value estimation vector for angle actions
            WSCN_Ang_NOP      = zeros(1,nAng);                %   Vector to store angle action ocurrences

            WSC_Ang         = angleUAV(fix(length(angleUAV)/2));       %   Angle initialization
            WSC_UAV = 0;                                    %   WSC initial value

            % Initialization for GD process
            Ang_Step = WSC_Ang;                             %   The same as the RL approach, to better compare the two approaches

            for i=1:nLoops
                dAB_R = dAB_R_V(i);
                
                % Exhaustive Search results
                [WSC_Max_Val, Ang_Max_Val, ~]       = optimalWSC_ZF(A, E, Rj, hj, dAB_R, gammaA, gammaJ, channelParam, angleUAV, nUAV, typeA );
                [WSC_Max_ValNOP, Ang_Max_ValNOP, ~] = optimalWSC_NOP(A, E, Rj, hj, dAB_R, gammaA, gammaJ, channelParam, angleUAV, nUAV, typeA );

                WSC_Max_ZF(iMC,i,iVm, iVa)       = WSC_Max_Val;
                Ang_Max_ZF_V(iMC,i,iVm, iVa)     = Ang_Max_Val;

                WSC_Max_NOP(iMC,i,iVm, iVa)      = WSC_Max_ValNOP;
                Ang_Max_NOP_V(iMC,i,iVm, iVa)    = Ang_Max_ValNOP;
                
                
                % dAB estimation and parameter computation
                dAB = normrnd(dAB_R,sigAB);                             %   CSI estimate

                %   RL iteration - ZF
                [WSCEst_Angle, WSCN_Angle]      = computeRL_UCB(WSCEst_Angle, hj, Rj, WSCN_Angle, angleUAV, ...
                                                    A, E, dAB, gammaA, gammaJ, c, channelParam, i, alpha, nUAV,1, typeA);

                %   RL iteration - NOP
                [WSCEst_Ang_NOP, WSCN_Ang_NOP]  = computeRL_UCB(WSCEst_Ang_NOP, hj, Rj, WSCN_Ang_NOP, angleUAV, ...
                                                    A, E, dAB, gammaA, gammaJ, c, channelParam, i, alpha, nUAV,3, typeA);
                
                %   GD iteration
                Ang_Step                        = computeGD(A, E, Rj, hj, dAB, gammaA, gammaJ, channelParam, alpha, Ang_Step, i, nUAV, typeA );

                %   True (Greedy) WSC calculations

                % ZF
                [~, Ang_RL_Ind]                 = max(WSCEst_Angle);
                Ang_RL                          = angleUAV(Ang_RL_Ind);
                UAVs                            = setNewPos_N(nUAV, Ang_RL, hj, Rj, typeA);
                WSC_RL(iMC,i,iVm, iVa)          = computeWSC_ZF_NUAV(A, E, UAVs, dAB_R, gammaA, gammaJ, channelParam );

                % NOP
                [~, Ang_NOP_Ind]                = max(WSCEst_Ang_NOP);
                Ang_NOP                         = angleUAV(Ang_NOP_Ind);
                UAVs                            = setNewPos_N(nUAV, Ang_NOP, hj, Rj, typeA);
                WSC_NOP(iMC,i,iVm, iVa)         = computeWSC_NOP_NUAV(A, E, UAVs, dAB_R, gammaA, gammaJ, channelParam );

                % GD
                UAVs                            = setNewPos_N(nUAV, Ang_Step, hj, Rj, typeA);
                WSC_GD(iMC,i,iVm, iVa)          = computeWSC_ZF_NUAV(A, E, UAVs, dAB_R, gammaA, gammaJ, channelParam );

                Ang_RL_V(iMC,i,iVm, iVa)        = Ang_RL;
                Ang_NOP_V(iMC,i,iVm, iVa)       = Ang_NOP;
                Ang_Step_V(iMC,i,iVm, iVa)      = Ang_Step;

                dists_AB(iMC,i,iVm, iVa)        = dAB_R;
            end
            
            % /*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*
        end
        % Time
        t1 = toc;
        if iMC*iVm == 1
            dt = t1;
        else
            dt = dt + alphat*(t1 - dt);
        end

        TT = dt*nMC*nMain;
        TTF = TT - ( (iMC-1)*nMain + iVm )*dt;

        TTF_S = rem(TTF,60);
        TTF_M = rem(fix(TTF/60),60);
        TTF_H = fix(fix(TTF/60)/60);

        fprintf('MC Loop: %i / %i\t\t %s: %.3f \t\t %s: %.3f \t\t  Time per Loop: %.2f s\t\t TTF: %i H %i M %.1f S \n',iMC,nMC,strMain,vecMain(iVm),strVals,vecVals(iVa),t1,TTF_H,TTF_M,TTF_S);
    end
    
end
% save(['TVT_Plan-',strMain, '-',strVals, '-AngT-', num2str(typeA), '-', verStr ])

%% h - sigmaAB
clc, clear, close all
hj_V        = 30;
sigmaAB_V   = [0.1, 1, 5, 10, 20];
verStr = 'PreTy-v00R.mat';

vecMain = hj_V;
vecVals = sigmaAB_V;
nMain = length(vecMain);
nVals = length(vecVals);
strMain = 'hj';
strVals = 'sigmaAB';

texMain = '$h_{\mathrm{J}}$';
texVals = '$\sigma_{\mathrm{AB}}$';

nMC = 1e3;
typeA = 2 ;         % 1: symmetric angles, 2: semicircle angles 
% ----------------------------------------------------------------------

% Environment parameters (Urban)
phi = 9.61;         % Environmental constant
omega = 0.16 ;      % Environmental constant
alpha = 0.3;        % Ground Path Loss exponent
alpha_AG = 0.3;     % Air-to-Ground Path Loss Exponent
ne_LOS = 1.0;       % Air-to-Ground LOS attenuation
ne_NLOS = 20;       % Air-to-Ground NLOS attenuation

% Channel parameters
m = 3;              % Number of parallel channels
sigma = 1/sqrt(2);  % Noise std dev of each component (real and imag) of every parallel channel
choice = 1;         % 0: Rayleigh channel, 1: Nakagami-m channel
Rs = 1;
channelParam =  [   phi,...
                    omega,...
                    alpha,...
                    alpha_AG,...
                    ne_LOS,...
                    ne_NLOS,...
                    Rs,...
                    m,...
                    sigma,...
                    choice];

% Positions ***************************************************************
    % Alice
A = [0,0,0];        %   Position of Alice (zero point)
gammaA = 100;       %   Alice Tx SNR

    % Bob
% sigAB = 1;          %   Unreliability of B's position

    % Eve
nR = 50;            %   Number of radial points
nTheta = 180;       %   Number of angular points
nE = nR*nTheta;     %   Number of Eves

rLow = 0.1;         %   Lowest radius of Eve
rHigh = 50;         %   Highest radius of Eve
thetaLow = 0;       %   Lowest angle of Eve
thetaHigh = 2*pi;     %   Highest radius of Eve

rangeR = linspace(rLow,rHigh,nR);                       % Points in Radial dimension
rangeTheta = linspace(thetaLow,thetaHigh,nTheta);       % Points in Angular dimension

thetat  =  repmat(rangeTheta,1,nR);
rt  = (repmat(rangeR',1,nTheta)).';

E = [rt(:).*cos(thetat(:)), rt(:).*sin(thetat(:)) , zeros(nR*nTheta,1)];             % Eves' position (rectangle coordinates)

% -------------------------------------------------------------------------
    % UAV

nAng = 10;                                          %   Angle discretization level (opening angle)  -> Number of Angle Actions
nUAV = 2;                                           %   Number of simultaneous UAVs
gammaJ = gammaA/nUAV;                           %   UAVs Jamming SNR
if typeA==1
    angleUAV    = linspace(0,2*pi/(nUAV-1),nAng);       %   Possible angle actions (opening angles)
elseif typeA==2
    angleUAV    = linspace(0,pi/(nUAV-1),nAng);
end

% hj = 70;                            % Fixed
Rj = 30;                            % Fixed

% -------------------------------------------------------------------------

% *************************************************************************

%   k-Armed Bandits
nLoops = 20;               % Number of loops for action choosing
initWSC = 0;                % Optimistic initial action values
c = 0.3;                    % Exploration parameter for UCB
alpha = 0.1;                % Step size (0: uniform average)

%   Performance Variables
dt = 0;
alphat = 0.2;

WSC_RL      = zeros(nMC,nLoops,nMain,nVals);
WSC_NOP     = zeros(nMC,nLoops,nMain,nVals);
WSC_GD      = zeros(nMC,nLoops,nMain,nVals);
WSC_Max_ZF  = zeros(nMC,nLoops,nMain,nVals);
WSC_Max_NOP = zeros(nMC,nLoops,nMain,nVals);

Ang_RL_V        = zeros(nMC,nLoops,nMain,nVals);
Ang_NOP_V       = zeros(nMC,nLoops,nMain,nVals);
Ang_Step_V      = zeros(nMC,nLoops,nMain,nVals);
Ang_Max_ZF_V    = zeros(nMC,nLoops,nMain,nVals);
Ang_Max_NOP_V   = zeros(nMC,nLoops,nMain,nVals);

dists_AB = zeros(nMC,nLoops,nMain,nVals);

% /*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*

for iMC =1:nMC
    
    % Bob's movement
    xo = 0;                     
    xf = 0;
    while abs(xo-xf)<=nLoops*1e-2   % Do not allow movements that are too short (such that each step is more than 0.01)
        xo = rHigh*rand();          % Initial position
        xf = rHigh*rand();          % Final position
    end
    dAB_R_V = linspace(xo,xf,nLoops);       % Total trajectory for all Loops
    
    
    for iVm = 1:nMain
        hj = vecMain(iVm);
        tic
        
        
        % Exhaustive Search results
        for i=1:nLoops
            % Bob's new position
            dAB_R = dAB_R_V(i);
            % Exhaustive Search results
            [WSC_Max_Val, Ang_Max_Val, ~]       = optimalWSC_ZF(A, E, Rj, hj, dAB_R, gammaA, gammaJ, channelParam, angleUAV, nUAV, typeA );
            [WSC_Max_ValNOP, Ang_Max_ValNOP, ~] = optimalWSC_NOP(A, E, Rj, hj, dAB_R, gammaA, gammaJ, channelParam, angleUAV, nUAV, typeA );

            WSC_Max_ZF(iMC,i,iVm, :)       = WSC_Max_Val;
            Ang_Max_ZF_V(iMC,i,iVm, :)     = Ang_Max_Val;

            WSC_Max_NOP(iMC,i,iVm, :)      = WSC_Max_ValNOP;
            Ang_Max_NOP_V(iMC,i,iVm, :)    = Ang_Max_ValNOP;
        end
        
        for iVa = 1:nVals
            sigAB = vecVals(iVa);
            % /*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*
                % Initialization for RL process
            WSCEst_Angle    = initWSC*ones(1,nAng);         %   Action value estimation vector for angle actions
            WSCN_Angle      = zeros(1,nAng);                %   Vector to store angle action ocurrences

            WSCEst_Ang_NOP    = initWSC*ones(1,nAng);         %   Action value estimation vector for angle actions
            WSCN_Ang_NOP      = zeros(1,nAng);                %   Vector to store angle action ocurrences

            WSC_Ang         = angleUAV(fix(length(angleUAV)/2));       %   Angle initialization
            WSC_UAV = 0;                                    %   WSC initial value

            % Initialization for GD process
            Ang_Step = WSC_Ang;                             %   The same as the RL approach, to better compare the two approaches

            for i=1:nLoops
                dAB_R = dAB_R_V(i);
                
                % dAB estimation and parameter computation
                dAB = normrnd(dAB_R,sigAB);                             %   CSI estimate

                %   RL iteration - ZF
                [WSCEst_Angle, WSCN_Angle]      = computeRL_UCB(WSCEst_Angle, hj, Rj, WSCN_Angle, angleUAV, ...
                                                    A, E, dAB, gammaA, gammaJ, c, channelParam, i, alpha, nUAV,1, typeA);

                %   RL iteration - NOP
                [WSCEst_Ang_NOP, WSCN_Ang_NOP]  = computeRL_UCB(WSCEst_Ang_NOP, hj, Rj, WSCN_Ang_NOP, angleUAV, ...
                                                    A, E, dAB, gammaA, gammaJ, c, channelParam, i, alpha, nUAV,3, typeA);
                
                %   GD iteration
                Ang_Step                        = computeGD(A, E, Rj, hj, dAB, gammaA, gammaJ, channelParam, alpha, Ang_Step, i, nUAV, typeA );

                %   True (Greedy) WSC calculations

                % ZF
                [~, Ang_RL_Ind]                 = max(WSCEst_Angle);
                Ang_RL                          = angleUAV(Ang_RL_Ind);
                UAVs                            = setNewPos_N(nUAV, Ang_RL, hj, Rj, typeA);
                WSC_RL(iMC,i,iVm, iVa)          = computeWSC_ZF_NUAV(A, E, UAVs, dAB_R, gammaA, gammaJ, channelParam );

                % NOP
                [~, Ang_NOP_Ind]                = max(WSCEst_Ang_NOP);
                Ang_NOP                         = angleUAV(Ang_NOP_Ind);
                UAVs                            = setNewPos_N(nUAV, Ang_NOP, hj, Rj, typeA);
                WSC_NOP(iMC,i,iVm, iVa)         = computeWSC_NOP_NUAV(A, E, UAVs, dAB_R, gammaA, gammaJ, channelParam );

                % GD
                UAVs                            = setNewPos_N(nUAV, Ang_Step, hj, Rj, typeA);
                WSC_GD(iMC,i,iVm, iVa)          = computeWSC_ZF_NUAV(A, E, UAVs, dAB_R, gammaA, gammaJ, channelParam );

                Ang_RL_V(iMC,i,iVm, iVa)        = Ang_RL;
                Ang_NOP_V(iMC,i,iVm, iVa)       = Ang_NOP;
                Ang_Step_V(iMC,i,iVm, iVa)      = Ang_Step;

                dists_AB(iMC,i,iVm, iVa)        = dAB_R;
            end

            
            
            % /*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*
        end
        % Time
        t1 = toc;
        if iMC*iVm == 1
            dt = t1;
        else
            dt = dt + alphat*(t1 - dt);
        end

        TT = dt*nMC*nMain;
        TTF = TT - ( (iMC-1)*nMain + iVm )*dt;

        TTF_S = rem(TTF,60);
        TTF_M = rem(fix(TTF/60),60);
        TTF_H = fix(fix(TTF/60)/60);

        fprintf('MC Loop: %i / %i\t\t %s: %.3f \t\t %s: %.3f \t\t  Time per Loop: %.2f s\t\t TTF: %i H %i M %.1f S \n',iMC,nMC,strMain,vecMain(iVm),strVals,vecVals(iVa),t1,TTF_H,TTF_M,TTF_S);
    end
    
end
% save(['TVT_Plan-',strMain, '-',strVals, '-AngT-', num2str(typeA), '-', verStr ])

%% m - gammaJ
clc, clear, close all
gammaJ_V    = [10, 50, 100, 500, 1000];
m_V      = [1, 3, 5];
verStr = 'PreTy-v03.mat';

vecMain = gammaJ_V;
vecVals = m_V;
nMain = length(vecMain);
nVals = length(vecVals);
strMain = 'gammaJ';
strVals = 'm';

texMain = '$\gamma_{\mathrm{J}}$';
texVals = '$m$';

nMC     = 500;
typeA   = 2 ;       % 1: symmetric angles, 2: semicircle angles 
% ----------------------------------------------------------------------
            
% Environment parameters (Urban)
phi         = 9.61;         % Environmental constant
omega       = 0.16 ;      % Environmental constant
alpha       = 0.3;        % Ground Path Loss exponent
alpha_AG    = 0.3;     % Air-to-Ground Path Loss Exponent
ne_LOS      = 1.0;       % Air-to-Ground LOS attenuation
ne_NLOS     = 20;       % Air-to-Ground NLOS attenuation

% Channel parameters
sigma = 1/sqrt(2);  % Noise std dev of each component (real and imag) of every parallel channel
choice = 1;         % 0: Rayleigh channel, 1: Nakagami-m channel
Rs = 1;
m = 3;              % Number of parallel channels
channelParam =  [   phi,...
                    omega,...
                    alpha,...
                    alpha_AG,...
                    ne_LOS,...
                    ne_NLOS,...
                    Rs,...
                    m,...
                    sigma,...
                    choice];

% Positions ***************************************************************
    % Alice
A = [0,0,0];        %   Position of Alice (zero point)
gammaA = 100;       %   Alice Tx SNR

    % Bob
sigAB = 1;          %   Unreliability of B's position

    % Eve
nR = 50;            %   Number of radial points
nTheta = 180;       %   Number of angular points
nE = nR*nTheta;     %   Number of Eves

rLow = 0.1;         %   Lowest radius of Eve
rHigh = 50;         %   Highest radius of Eve
thetaLow = 0;       %   Lowest angle of Eve
thetaHigh = 2*pi;     %   Highest radius of Eve

rangeR = linspace(rLow,rHigh,nR);                       % Points in Radial dimension
rangeTheta = linspace(thetaLow,thetaHigh,nTheta);       % Points in Angular dimension

thetat  =  repmat(rangeTheta,1,nR);
rt  = (repmat(rangeR',1,nTheta)).';

E = [rt(:).*cos(thetat(:)), rt(:).*sin(thetat(:)) , zeros(nR*nTheta,1)];             % Eves' position (rectangle coordinates)

% -------------------------------------------------------------------------
    % UAV

nAng = 10;                                          %   Angle discretization level (opening angle)  -> Number of Angle Actions

nUAV = 2;                                            %   Number of simultaneous UAVs
gammaJ = gammaA/nUAV;
% angleUAV    = linspace(0,2*pi/(nUAV-1),nAng);       %   Possible angle actions (opening angles)

hj = 30;                            % Fixed
Rj = 40;                            % Fixed

% -------------------------------------------------------------------------

% *************************************************************************

%   k-Armed Bandits
nLoops = 20;               % Number of loops for action choosing
initWSC = 0;                % Optimistic initial action values
c = 0.3;                    % Exploration parameter for UCB
alpha = 0.1;                % Step size (0: uniform average)

%   Performance Variables
dt = 0;
alphat = 0.2;

WSC_RL      = zeros(nMC,nLoops,nMain,nVals);
WSC_NOP     = zeros(nMC,nLoops,nMain,nVals);
WSC_GD      = zeros(nMC,nLoops,nMain,nVals);
WSC_Max_ZF  = zeros(nMC,nLoops,nMain,nVals);
WSC_Max_NOP = zeros(nMC,nLoops,nMain,nVals);

Ang_RL_V        = zeros(nMC,nLoops,nMain,nVals);
Ang_NOP_V       = zeros(nMC,nLoops,nMain,nVals);
Ang_Step_V      = zeros(nMC,nLoops,nMain,nVals);
Ang_Max_ZF_V    = zeros(nMC,nLoops,nMain,nVals);
Ang_Max_NOP_V   = zeros(nMC,nLoops,nMain,nVals);

dists_AB = zeros(nMC,nLoops,nMain,nVals);

% /*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*
% rng(1)
for iMC =1:nMC
    
    % Bob's movement
    xo = 0;                     
    xf = 0;
    while abs(xo-xf)<=nLoops*1e-2   % Do not allow movements that are too short (such that each step is more than 0.01)
        xo = rHigh*rand();          % Initial position
        xf = rHigh*rand();          % Final position
    end
    dAB_R_V = linspace(xo,xf,nLoops);       % Total trajectory for all Loops
    
    for iVm = 1:nMain
        gammaJ = vecMain(iVm);
        tic
        for iVa = 1:nVals
            
            m = vecVals(iVa);              % Number of parallel channels
            channelParam =  [   phi,...
                                omega,...
                                alpha,...
                                alpha_AG,...
                                ne_LOS,...
                                ne_NLOS,...
                                Rs,...
                                m,...
                                sigma,...
                                choice];
            
            
%             nUAV = vecVals(iVa);
            
%             gammaJ = gammaA;                           %   UAVs Jamming SNR
            if typeA==1
                angleUAV    = linspace(0,2*pi/(nUAV-1),nAng);       %   Possible angle actions (opening angles)
            elseif typeA==2
                angleUAV    = linspace(0,pi/(nUAV-1),nAng);
            end
            
%             
%             angleUAV    = linspace(0.5*pi/(nUAV-1),pi/(nUAV-1),nAng);
            
                        
            % /*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*
                % Initialization for RL process
            WSCEst_Angle    = initWSC*ones(1,nAng);         %   Action value estimation vector for angle actions
            WSCN_Angle      = zeros(1,nAng);                %   Vector to store angle action ocurrences

            WSCEst_Ang_NOP    = initWSC*ones(1,nAng);         %   Action value estimation vector for angle actions
            WSCN_Ang_NOP      = zeros(1,nAng);                %   Vector to store angle action ocurrences

            WSC_Ang         = angleUAV(fix(length(angleUAV)/2));       %   Angle initialization
            WSC_UAV = 0;                                    %   WSC initial value

            % Initialization for GD process
            Ang_Step = WSC_Ang;                             %   The same as the RL approach, to better compare the two approaches

            for i=1:nLoops
                dAB_R = dAB_R_V(i);
                
                % Exhaustive Search results
%                 fprintf('Optimal ZF\n\n')
                [WSC_Max_Val, Ang_Max_Val, ~]       = optimalWSC_ZF(A, E, Rj, hj, dAB_R, gammaA, gammaJ, channelParam, angleUAV, nUAV, typeA );
%                 fprintf('Optimal NOP\n\n')
                [WSC_Max_ValNOP, Ang_Max_ValNOP, ~] = optimalWSC_NOP(A, E, Rj, hj, dAB_R, gammaA, gammaJ, channelParam, angleUAV, nUAV, typeA );

                WSC_Max_ZF(iMC,i,iVm, iVa)       = WSC_Max_Val;
                Ang_Max_ZF_V(iMC,i,iVm, iVa)     = Ang_Max_Val;

                WSC_Max_NOP(iMC,i,iVm, iVa)      = WSC_Max_ValNOP;
                Ang_Max_NOP_V(iMC,i,iVm, iVa)    = Ang_Max_ValNOP;
                
                
                % dAB estimation and parameter computation
                dAB = normrnd(dAB_R,sigAB);                             %   CSI estimate

                %   RL iteration - ZF
                [WSCEst_Angle, WSCN_Angle]      = computeRL_UCB(WSCEst_Angle, hj, Rj, WSCN_Angle, angleUAV, ...
                                                    A, E, dAB, gammaA, gammaJ, c, channelParam, i, alpha, nUAV,1, typeA );

                %   RL iteration - NOP
                [WSCEst_Ang_NOP, WSCN_Ang_NOP]  = computeRL_UCB(WSCEst_Ang_NOP, hj, Rj, WSCN_Ang_NOP, angleUAV, ...
                                                    A, E, dAB, gammaA, gammaJ, c, channelParam, i, alpha, nUAV,3, typeA);
                
                %   GD iteration
                Ang_Step                        = computeGD(A, E, Rj, hj, dAB, gammaA, gammaJ, channelParam, alpha, Ang_Step, i, nUAV, typeA );

                %   True (Greedy) WSC calculations

                % ZF
                [~, Ang_RL_Ind]                 = max(WSCEst_Angle);
                Ang_RL                          = angleUAV(Ang_RL_Ind);
                UAVs                            = setNewPos_N(nUAV, Ang_RL, hj, Rj, typeA);
                WSC_RL(iMC,i,iVm, iVa)          = computeWSC_ZF_NUAV(A, E, UAVs, dAB_R, gammaA, gammaJ, channelParam );

                % NOP
                [~, Ang_NOP_Ind]                = max(WSCEst_Ang_NOP);
                Ang_NOP                         = angleUAV(Ang_NOP_Ind);
                UAVs                            = setNewPos_N(nUAV, Ang_NOP, hj, Rj, typeA);
                WSC_NOP(iMC,i,iVm, iVa)         = computeWSC_NOP_NUAV(A, E, UAVs, dAB_R, gammaA, gammaJ, channelParam );

                % GD
                UAVs                            = setNewPos_N(nUAV, Ang_Step, hj, Rj, typeA);
                WSC_GD(iMC,i,iVm, iVa)          = computeWSC_ZF_NUAV(A, E, UAVs, dAB_R, gammaA, gammaJ, channelParam );

                Ang_RL_V(iMC,i,iVm, iVa)        = Ang_RL;
                Ang_NOP_V(iMC,i,iVm, iVa)       = Ang_NOP;
                Ang_Step_V(iMC,i,iVm, iVa)      = Ang_Step;

                dists_AB(iMC,i,iVm, iVa)        = dAB_R;
            end

            
            
            % /*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*
        end
        % Time
        t1 = toc;
        if iMC*iVm == 1
            dt = t1;
        else
            dt = dt + alphat*(t1 - dt);
        end

        TT = dt*nMC*nMain;
        TTF = TT - ( (iMC-1)*nMain + iVm )*dt;

        TTF_S = rem(TTF,60);
        TTF_M = rem(fix(TTF/60),60);
        TTF_H = fix(fix(TTF/60)/60);

        fprintf('MC Loop: %i / %i\t\t %s: %.3f \t\t %s: %.3f \t\t  Time per Loop: %.2f s\t\t TTF: %i H %i M %.1f S \n',iMC,nMC,strMain,vecMain(iVm),strVals,vecVals(iVa),t1,TTF_H,TTF_M,TTF_S);
    end
    
end
% save(['TVT_Plan-',strMain, '-',strVals, '-AngT-', num2str(typeA), '-', verStr ])

%%