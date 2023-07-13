# 🧠 HelloNeuralNetworks
Summer project of 2023, a first step into the world of AI, all in C of course ;)

I hope to be able to show this off to my peers by the end of the summer.

# 📜 Objectives :
* create a matrix operations header ✅
* create some sort of csv loader
* create a text user interface to visualize data ✅
* create a neural network ✅
* create a system to train the network ✅ (can be trained, backpropagation would make it better)
* train the neural network on data to recognize drawn numbers
* create an interface to interact with the neural network

# 🐧 Compiling on Linux :

## ⚠️DISCLAIMER⚠️ : 

the project is currently under development, therefore, compilation might change, 
I'm leaving the steps that will be needed to compile the project once it is completed,
though they arent exactly the same today (SDL2 currently is not needed)

## END OF DISCLAIMER

Compiling on Linux is trivial, on most distributions you should only need to install the SDL2 library
```
Fedora : "SDL2-devel"
Arch   : "sdl2"

```
then simply go to the project directory and type ```make release``` in your command prompt.
An executable should have been created in the **./build/release** directory if all went well.

# 🪟 Compiling on Windows

I have not currently set up the project to compile on Windows since doing so is complicated,
though you *might* be able to if you have a terminal emulator that supports ANSI escape codes
moreover SDL is a cross platform library so it shouldn't be impossible.
