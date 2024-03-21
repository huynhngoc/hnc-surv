Bootstrap: docker
From: deoxys-survival
Stage: build

%post
    pip install shap
    pip install visualkeras