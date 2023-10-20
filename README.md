# How To Compile:
I use Visual Studio with Libtorch Framework to compile my project

So I use "D:\Program Files (x86)\VS\MSBuild\Current\Bin\amd64\MSBuild.exe" ./Project1x.vcxproj /t:build /p:Configuration=Release /p:Platform=x64" command to compile.

# How to run: 
we suppose the compiled program name is P

"P Index A " command to run the Index on dataset A

"P CE A " command to run the CE on dataset A

# Code Structure

Code.cpp: The CardIndex code.

Query.py: generate queries.

Project1x.sln\vcxproj: The configuration in my Visual Studio.

compile.cpp: the example compile command in my computer.

# The dataset we use:

power: Individual household electric power consumption data set. http://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption

DMV: https://catalog.data.gov/dataset/vehicle-snowmobile-and-boat-registrations

OSM: https://www.openstreetmap.org