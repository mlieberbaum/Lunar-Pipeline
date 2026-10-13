function buildLunarTextureAuto(orbiterFile, seamlessFile, morningFile, eveningFile, outputFile, quiet, sourceWorkers)
% Development processor. Supports raw IMG or existing uint16 Kaguya PNGs.
% Estimates alignment and strip masks locally; requires visual review.
arguments
    orbiterFile (1,1) string
    seamlessFile (1,1) string
    morningFile (1,1) string
    eveningFile (1,1) string
    outputFile (1,1) string
    quiet (1,1) logical = false
    sourceWorkers (1,1) double {mustBeMember(sourceWorkers, [1 2 3])} = 3
end
folder = fileparts(mfilename('fullpath'));
pyrun("import sys; sys.path.insert(0, helperFolder)", helperFolder=char(folder));
pyrunfile(fullfile(folder, 'lunar_texture_auto.py'), ...
    orbiterFile=char(orbiterFile), seamlessFile=char(seamlessFile), ...
    morningFile=char(morningFile), eveningFile=char(eveningFile), ...
    outputFile=char(outputFile), quiet=quiet, sourceWorkers=sourceWorkers);
end
