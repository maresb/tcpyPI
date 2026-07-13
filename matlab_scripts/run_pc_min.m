% run_pc_min.m
%
% Demonstrate and validate matlab_scripts/pc_min.m under GNU Octave.
%
% pc_min.m is Kerry Emanuel's MATLAB potential-intensity (BE02/pcmin)
% algorithm.  This driver runs it on a handful of real soundings taken from
% data/sample_data.nc and checks that the outputs still match the reference
% PMIN/VMAX/TO/LNB/IFL values that the original MATLAB code stored in that
% file.  It uses only core Octave features (no toolboxes / OF packages), so it
% doubles as a smoke test that pc_min.m runs unmodified under Octave.
%
% Run from the repository root with:
%     octave-cli --no-gui matlab_scripts/run_pc_min.m
% or, using the reproducible pixi environment:
%     pixi run pc-min-octave
%
% Exits with a non-zero status if any output drifts beyond tolerance, so it is
% safe to use in CI.

1;   % force Octave/MATLAB to treat this file as a script, not a function

% Locate pc_min.m relative to this script so the CWD does not matter.
this_dir = fileparts(mfilename('fullpath'));
if isempty(this_dir)
  this_dir = pwd;
end
addpath(this_dir);

% Pressure grid (hPa); lowest index = surface (highest pressure).
P = [ ...
       1000.000000   975.000000   950.000000   925.000000   900.000000   875.000000   850.000000   825.000000 ...
        800.000000   775.000000   750.000000   725.000000   700.000000   650.000000   600.000000   550.000000 ...
        500.000000   450.000000   400.000000   350.000000   300.000000   250.000000   200.000000   150.000000 ...
        100.000000    70.000000    50.000000    40.000000    30.000000    20.000000    10.000000
    ];

% ---------------------------------------------------------------------------
% Sample soundings from data/sample_data.nc (2004 MERRA2 monthly means).
% For each site: sst (C), psl (hPa), T (C) and R (g/kg) on the P grid, plus
% the reference PMIN/VMAX/TO/LNB/IFL that the original MATLAB pcmin algorithm
% (via reference_calculations.m) stored in sample_data.nc.
% ---------------------------------------------------------------------------
samples = {};

s = struct();
s.name = 'Atlantic MDR      (Sep, 15.0N  50.0W)';
s.sst  = 28.283716;  s.psl = 1012.959480;
s.T = [ ...
         26.128353    23.953175    21.893749    20.432191    19.511166    18.645990    17.632402    16.438769 ...
         15.127264    13.676948    12.103871    10.456784     8.718429     5.414111     2.166769    -1.504595 ...
         -5.632913   -10.251407   -15.726427   -22.314080   -30.889780   -41.284452   -53.950385   -69.107242 ...
        -76.967310   -70.097407   -63.926093   -60.888493   -57.378079   -52.397561   -43.602938
    ];
s.R = [ ...
         16.818064    16.480417    16.082236    15.071074    13.235643    11.457873     9.959320     8.801524 ...
          7.866860     7.224431     6.839392     6.269362     5.598768     4.529739     3.419680     2.608250 ...
          1.697215     1.157443     0.739657     0.503774     0.289018     0.153317     0.060470     0.012051 ...
          0.003218     0.002754     0.002493     0.002492     0.002565     0.002703     0.002865
    ];
s.exp = struct("PMIN",918.176040,"VMAX",74.447840,"TO",197.111330,"LNB",95.944749,"IFL",1);
samples{end+1} = s;

s = struct();
s.name = 'W Pacific         (Sep, 10.0N 130.0E)';
s.sst  = 29.167596;  s.psl = 1010.014605;
s.T = [ ...
         26.635089    24.439779    22.471646    21.476805    20.645548    19.458651    18.136958    16.784823 ...
         15.392220    13.941792    12.404695    10.768142     9.030299     5.745629     2.602114    -0.999998 ...
         -5.099399    -9.727772   -15.289821   -21.248207   -29.819932   -39.974820   -53.325972   -69.347862 ...
        -78.175791   -70.589556   -64.491015   -61.531487   -57.938858   -53.058540   -44.524513
    ];
s.R = [ ...
         17.845321    17.444604    16.756477    14.932384    13.374116    12.153048    11.249268    10.441693 ...
          9.839789     9.190867     8.624877     8.180589     7.619224     6.407971     5.097867     3.859148 ...
          2.957396     2.343197     1.738339     1.165871     0.655064     0.282813     0.075559     0.012212 ...
          0.003350     0.002777     0.002422     0.002417     0.002507     0.002663     0.002826
    ];
s.exp = struct("PMIN",894.581279,"VMAX",78.338820,"TO",197.798844,"LNB",88.829895,"IFL",1);
samples{end+1} = s;

s = struct();
s.name = 'S Indian Ocean    (Jan, 15.0S  90.0E)';
s.sst  = 27.367181;  s.psl = 1012.482323;
s.T = [ ...
         25.009478    22.824735    20.662328    18.575255    16.740183    15.559029    15.428495    15.183864 ...
         14.198928    12.954978    11.610435    10.356337     9.166229     6.787473     3.863743     0.416445 ...
         -3.821116    -8.792563   -14.661822   -21.637744   -29.947092   -40.297253   -52.811688   -67.203503 ...
        -81.151240   -80.459724   -68.762487   -63.417584   -59.019030   -53.450844   -43.209440
    ];
s.R = [ ...
         13.958838    13.582968    13.405044    13.173239    12.714999    11.653858     9.256485     7.074956 ...
          5.862636     4.877026     4.159063     3.521957     2.965496     1.947308     1.323431     0.871043 ...
          0.682382     0.524628     0.322145     0.210586     0.137603     0.095351     0.045497     0.011238 ...
          0.001735     0.002033     0.002668     0.002659     0.002661     0.002712     0.002897
    ];
s.exp = struct("PMIN",943.992041,"VMAX",71.297657,"TO",192.089659,"LNB",96.056538,"IFL",1);
samples{end+1} = s;

s = struct();
s.name = 'E Pacific         (Mar, 12.5N  80.0W)';
s.sst  = 26.971338;  s.psl = 1012.096387;
s.T = [ ...
         25.592061    23.458084    21.343195    19.288628    17.523839    16.669552    16.618246    15.673410 ...
         14.509557    13.547764    12.645929    11.535149    10.151511     7.047201     3.729247     0.015416 ...
         -4.261430    -9.227689   -15.002352   -22.109809   -30.623584   -40.961208   -53.159702   -66.928256 ...
        -81.199993   -78.847462   -68.036613   -63.499057   -57.621676   -51.092295   -41.653157
    ];
s.R = [ ...
         15.331695    14.929051    14.634244    14.215381    13.439885    11.469991     9.024408     7.965374 ...
          7.072413     5.761228     4.606061     3.942692     3.501126     2.644428     1.782784     1.125221 ...
          0.658254     0.388708     0.298001     0.235115     0.184613     0.124696     0.054366     0.010364 ...
          0.001751     0.002242     0.002536     0.002599     0.002615     0.002622     0.002917
    ];
s.exp = struct("PMIN",952.575309,"VMAX",66.245400,"TO",192.055909,"LNB",98.649515,"IFL",1);
samples{end+1} = s;

s = struct();
s.name = 'Subtropical Atl   (Sep, 20.0N  40.0W)';
s.sst  = 27.592920;  s.psl = 1015.488817;
s.T = [ ...
         25.959912    23.832336    21.734587    19.824570    18.350027    17.637242    17.674987    17.339124 ...
         16.117772    14.602834    12.862236    10.960282     8.933380     4.946905     1.075030    -2.621009 ...
         -6.505909   -11.061310   -16.616452   -23.345407   -31.781772   -42.000687   -54.262210   -68.741657 ...
        -75.332438   -69.245163   -63.485284   -60.314563   -56.508453   -51.274812   -42.545047
    ];
s.R = [ ...
         15.956751    15.592541    15.273828    14.735097    13.698361    11.810206     9.209835     7.253412 ...
          6.581720     6.142732     5.805854     5.488160     5.131750     4.527318     3.705721     2.707881 ...
          1.857075     1.226028     0.884325     0.659732     0.419868     0.180471     0.054482     0.011273 ...
          0.003540     0.002731     0.002554     0.002564     0.002637     0.002760     0.002919
    ];
s.exp = struct("PMIN",938.150715,"VMAX",70.245448,"TO",198.709737,"LNB",106.768353,"IFL",1);
samples{end+1} = s;

% Absolute tolerances for the comparison against the stored MATLAB reference.
% Octave and MATLAB differ only at the level of iterative round-off here.
tol = struct('PMIN', 0.05, 'VMAX', 0.05, 'TO', 0.05, 'LNB', 0.10);

if exist('OCTAVE_VERSION', 'builtin')
  fprintf('Running pc_min.m under GNU Octave %s\n\n', OCTAVE_VERSION);
else
  fprintf('Running pc_min.m under MATLAB %s\n\n', version());
end

hdr = '%-38s | %8s %8s | %8s %8s | %8s %8s | %8s %8s | %3s\n';
fprintf(hdr, 'site', 'PMIN', 'ref', 'VMAX', 'ref', 'TO', 'ref', 'LNB', 'ref', 'IFL');
fprintf('%s\n', repmat('-', 1, 118));

failures = 0;
for k = 1:numel(samples)
  s = samples{k};
  [PMIN, VMAX, TO, LNB, IFL] = pc_min(s.sst, s.psl, P, s.T, s.R);

  fprintf('%-38s | %8.2f %8.2f | %8.2f %8.2f | %8.2f %8.2f | %8.2f %8.2f | %3d\n', ...
         s.name, PMIN, s.exp.PMIN, VMAX, s.exp.VMAX, TO, s.exp.TO, LNB, s.exp.LNB, IFL);

  checks = { 'PMIN', PMIN, s.exp.PMIN, tol.PMIN;
             'VMAX', VMAX, s.exp.VMAX, tol.VMAX;
             'TO',   TO,   s.exp.TO,   tol.TO;
             'LNB',  LNB,  s.exp.LNB,  tol.LNB };
  for j = 1:size(checks, 1)
    d = abs(checks{j, 2} - checks{j, 3});
    if d > checks{j, 4}
      failures = failures + 1;
      fprintf('    MISMATCH %-4s got %.6f expected %.6f (|d|=%.4g > %.4g)\n', ...
             checks{j, 1}, checks{j, 2}, checks{j, 3}, d, checks{j, 4});
    end
  end
  if IFL ~= s.exp.IFL
    failures = failures + 1;
    fprintf('    MISMATCH IFL  got %d expected %d\n', IFL, s.exp.IFL);
  end
end

fprintf('\n');
if failures > 0
  error('pc_min:validation', '%d output(s) drifted beyond tolerance', failures);
else
  fprintf('OK: pc_min.m ran under Octave and matched the MATLAB reference for all %d soundings.\n', ...
         numel(samples));
end
