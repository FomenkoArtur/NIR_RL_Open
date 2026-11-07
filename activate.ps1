$env:NIR_DEVICE        = "cpu"
$env:NIR_ENVS          = "32"
$env:OMP_NUM_THREADS       = "2"
$env:MKL_NUM_THREADS       = "2"
$env:OPENBLAS_NUM_THREADS  = "2"
$env:NUMEXPR_NUM_THREADS   = "2"
Write-Host "NIR_RL: device=cpu, envs=32, pytest threads=2" -ForegroundColor Green