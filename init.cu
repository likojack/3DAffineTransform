#include "luaT.h"
#include "THC.h"

#include "utils.c"

#include "BilinearSamplerThreeD.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libcustntd(lua_State *L);

int luaopen_libcustntd(lua_State *L)
{
  lua_newtable(L);
  cunn_BilinearSamplerThreeD_init(L);

  return 1;
}
