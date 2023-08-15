# 🧠 HelloNeuralNetworks
Summer project of 2023, a first step into the world of AI, all in C of course ;)

I hope to be able to show this off to my peers by the end of the summer.

# 📜 Objectives :
* create a matrix operations header ✅
* create some sort of csv/png loader ✅
* create a text user interface to visualize data ✅
* create a neural network ✅
* create a system to train the network ✅
* create a rate optimizer to further enhance learning (might do)
* train the neural network on data to recognize drawn numbers
* create an interface to interact with the neural network (WIP)

# 📸 UI :
The UI is a simple resizable window with a FPS counter.

The image output of the neural network is rendered inside the window like so :

![image](https://github.com/clemdemort/HelloNeuralNetworks/assets/62178977/f20f4e5e-fafd-4cf4-8599-4fb2cfc72b97)

# 🎮 Controls :

There are currently a few keybinds you can use 

* SPACE : pauses/unpauses the learning by the default the program is paused
* R : randomizes the neural network (starts learning all over again)
* UP : increases output quality
* Down : decreases output quality

# 🖥️ Launch options :

you can add a .png file as an option to change the image being used.

# 🐧 Compiling on Linux :

Compiling on Linux is trivial, on most distributions you should only need to install the SDL2 and the libpng libraries
```
Fedora : "SDL2-devel"
Arch   : "sdl2"
Fedora : "libpng"
Arch   : "libpng"
```
then simply go to the project directory and type ```make release``` in your command prompt.
An executable should have been created in the **./build/release** directory if all went well.

# 🪟 Compiling on Windows

I have not currently set up the project to compile on Windows since doing so is complicated,
though you *might* be able to if you have a terminal emulator that supports ANSI escape codes
moreover SDL is a cross platform library so it shouldn't be impossible.
