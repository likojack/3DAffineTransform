#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

#include "generic/BilinearSamplerThreeD.c"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libstntd(lua_State *L);

int luaopen_libstntd(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setglobal(L, "stntd");

  nn_FloatBilinearSamplerThreeD_init(L);

  nn_DoubleBilinearSamplerThreeD_init(L);

  return 1;
}
