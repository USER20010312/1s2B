@echo off
"D:\Program Files (x86)\VS\MSBuild\Current\Bin\amd64\MSBuild.exe" ./Project1x.vcxproj /t:build /p:Configuration=Release /p:Platform=x64
set "source=C:\Users\dell\source\repos\Project1x\x64\Release\Porject1x.exe"
set "destination=C:\Users\dell\source\repos\Project1x"
move "%source%" "%destination%"

Porject1x.exe CE power
Porject1x.exe CE DMV
Porject1x.exe CE OSM
Porject1x.exe Index power
Porject1x.exe Index DMV
Porject1x.exe Index OSM 

