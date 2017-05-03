package = "stntd"
version = "scm-1"

source = {
   url = "https://github.com/likojack/3DAffineTransform.git",
}

description = {
   summary = "3D Affine Transform layers for Torch",
   detailed = [[
   ]],
   homepage = "https://github.com/qassemoquab/stnbhwd",
   license = "MIT"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}
