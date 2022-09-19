function SOP = SOP_NakagamiM_N(beta, eta, OmegaB, OmegaE, mB, mE ,choice)
    % SOP computation for Nakagami-m ground channels
    
    switch choice
    case 1
        term1 =  (exp(-beta./OmegaB)) ./ ( gamma(mE).*( (1+ (OmegaE.*eta./OmegaB) ).^mE ) ) ;
        sumT = 0;
        for n = 0:mB-1
            term3 = 0;
            for k = 0:n
                term3 = term3 + (  nchoosek(n,k)*(( eta./( (beta./OmegaE).*(1+ (OmegaE.*eta./OmegaB) ) ) ).^k)*gamma(mE+k)  );
            end
            sumT = sumT + ((beta./OmegaB).^n).*(term3/(gamma(n+1)));
        end
        SOP = 1 - term1.*sumT;
    end
    SOP(SOP<0)=0;
end
