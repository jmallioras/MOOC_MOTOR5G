%% Define the Phased Antenna Array (Uniform Linear Array)

% Set the operating frequency
frequency = 27e9;

% Create a custom radiation pattern for the elements. This modification 
% makes the overall pattern more directive and reduces back lobes and the
% overall rear side of the antenna radiation pattern.

% Create azimuth and elevation vectors
azvec = -180:180; % Azimuth angles (deg)
elvec = -90:90;   % Elevation angles (deg)

% Generate a meshgrid of azimuth and elevation angles
[az, el] = meshgrid(azvec, elvec);

% Define custom pattern parameters
SLA = 30;       % Maximum side-lobe level attenuation (dB)
az3dB = 65;     % 3 dB beamwidth in azimuth (deg)
el3dB = 65;     % 3 dB beamwidth in elevation (deg)
tilt = 0;       % Elevation tilt

% Compute azimuth and elevation magnitude patterns
azMagPattern = -min(12 * (az / az3dB).^2, SLA);
elMagPattern = -min(12 * ((el - tilt) / el3dB).^2, SLA);

% Combine the magnitude patterns to form the total pattern
combinedMagPattern = -min(-(azMagPattern + elMagPattern), SLA); % Relative antenna gain (dB)

% Create a custom antenna element with the defined pattern
antennaElement = phased.CustomAntennaElement('MagnitudePattern', combinedMagPattern);

% Set up the ULA with 16 custom elements
nElements = 16;
lambda = physconst('lightspeed') / frequency; % Wavelength (λ)
ula = phased.ULA('NumElements', nElements, 'ElementSpacing', lambda / 2, ...
    'Element', antennaElement);


%% Import map
viewer = siteviewer('Basemap', 'openstreetmap', 'Buildings', 'thessaloniki.osm');

%% Clear any existing map data
clearMap(viewer);

% Transmitter site (Base Station)
tx = txsite('Name', 'BaseStation', 'Latitude', 40.627478, 'Longitude', 22.952896, ...
    'Antenna', ula, 'AntennaHeight', 3, ...
    'TransmitterPower', 2, ...
    'TransmitterFrequency', frequency);

% Set the antenna tilt angles
AZIMUTH_TILT = -90;
ELEVATION_TILT = 0;
tx.AntennaAngle = [AZIMUTH_TILT, ELEVATION_TILT];

% Show the pattern in 3D space
pattern(tx, 'Transparency', 0.6);

% Show transmitter on the map
show(tx);


%% Create a random DOA scenario

% Receiver sites
% Feel free to experiment with the coordinates of the desired user and 
% interferences and test various scenarios. Make sure that all users lie
% within the 30-150 angular sector as shown in lesson 2.

SOI = rxsite('Name', 'MobileReceiver', 'Latitude', 40.626458, 'Longitude', 22.953767);
show(SOI,"Icon","desired.png");

SOA1 = rxsite('Name', 'MobileReceiver', 'Latitude', 40.625970, 'Longitude',22.952462);
show(SOA1,"Icon","interference1.png");

SOA2 = rxsite('Name', 'MobileReceiver', 'Latitude', 40.625721, 'Longitude',22.953096);
show(SOA2,"Icon","interference2.png");

%% Define raytracing propagation model
rtpm = propagationModel("raytracing", ...
            "Method","sbr", ...
            "MaxNumReflections",0, ...
            "MaxNumDiffractions",0,...
            "AngularSeparation","medium",...
            "BuildingsMaterial","concrete", ...
            "TerrainMaterial","concrete");


% Get DoAs of SoAs
SOI_ray = raytrace(tx,SOI,rtpm);
SOA1_ray = raytrace(tx,SOA1,rtpm);
SOA2_ray = raytrace(tx,SOA2,rtpm);

% Show these rays
raytrace(tx,SOI,rtpm)
raytrace(tx,SOA1,rtpm)
raytrace(tx,SOA2,rtpm)

% Get the DOAs and allign them according to the antenna tilt
DOA_SOI = SOI_ray{1}.AngleOfDeparture;
DOA_SOI = [wrapTo180(DOA_SOI(1)-tx.AntennaAngle(1))];
DOA_SOA1 = SOA1_ray{1}.AngleOfDeparture;
DOA_SOA1 = [wrapTo180(DOA_SOA1(1)-tx.AntennaAngle(1))];
DOA_SOA2 = SOA2_ray{1}.AngleOfDeparture;
DOA_SOA2 = [wrapTo180(DOA_SOA2(1)-tx.AntennaAngle(1))];


% In Lesson 2, we saw that we only utilize the DoA angle "theta" for ULA 
% beamforming. This angle was called "elevation angle" given the vertical 
% topology in which the antenna presented. In the context of the current 
% implementation, the antenna is placed horizontally, in parallel with the 
% ground plane. Therefore, the same angle measurement, now expresses 
% the "azimuth angle" since, by definition, azimuth tells you what
% direction to face and elevation tells you how high up in the sky to look.


% Transform DoA azimuth from [-90, 90] to [0, 180] for the NNs and NSB
conv_SOI = DOA_SOI+90;
conv_SOA1 = DOA_SOA1+90;
conv_SOA2 = DOA_SOA2+90;

% Create a unified list with azimuth only angles
doas = [conv_SOI, conv_SOA1, conv_SOA2];
disp("DOAS:"+doas)
%% Get the NSB weights
wNSB = NSB(doas, nElements);
wFFNN = FFNN(doas);
wLSTM = LSTM(doas);

% Apply NSB weights
tx.Antenna.Taper = conj(wNSB);
pattern(tx, 'Transparency', 0.6);

% Show rays after NSB
raytrace(tx,SOI,rtpm)
raytrace(tx,SOA1,rtpm)
raytrace(tx,SOA2,rtpm)
ss = sigstrength(SOI,tx,rtpm);
ss1 = sigstrength(SOA1,tx,rtpm);
ss2 = sigstrength(SOA2,tx,rtpm);
sir = 10*log10(10^(ss/10)/(10^(ss1/10)+10^(ss2/10)));
disp("(NSB) Desired User Received power: " + ss + " dBm")
disp("(NSB) Received power interference 1: " + ss1 + " dBm | 2: " + ss2 +" dBm")
disp("(NSB) SIR: " + sir + " dBm")

%% Apply FFNN weights
tx.Antenna.Taper = conj(wFFNN);
pattern(tx, 'Transparency', 0.6);

% Show these rays after FFNN
raytrace(tx,SOI,rtpm)
raytrace(tx,SOA1,rtpm)
raytrace(tx,SOA2,rtpm)
ss = sigstrength(SOI,tx,rtpm);
ss1 = sigstrength(SOA1,tx,rtpm);
ss2 = sigstrength(SOA2,tx,rtpm);
sir = 10*log10(10^(ss/10)/(10^(ss1/10)+10^(ss2/10)));
disp("(FFNN) Desired User Received power: " + ss + " dBm")
disp("(FFNN) Received power interference 1: " + ss1 + " dBm | 2: " + ss2 +" dBm")
disp("(FFNN) SIR: " + sir + " dBm")

%% Apply LSTM weights
tx.Antenna.Taper = conj(wLSTM);
pattern(tx, 'Transparency', 0.6);

% Show these rays after LSTM
raytrace(tx,SOI,rtpm)
raytrace(tx,SOA1,rtpm)
raytrace(tx,SOA2,rtpm)
ss = sigstrength(SOI,tx,rtpm);
ss1 = sigstrength(SOA1,tx,rtpm);
ss2 = sigstrength(SOA2,tx,rtpm);
sir = 10*log10(10^(ss/10)/(10^(ss1/10)+10^(ss2/10)));
disp("(LSTM) Desired User Received power: " + ss + " dBm")
disp("(LSTM) Received power interference 1: " + ss1 + " dBm | 2: " + ss2 +" dBm")
disp("(LSTM) SIR: " + sir + " dBm")

