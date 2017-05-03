#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/BilinearSamplerThreeD.c"
#else

#include <stdbool.h>
#include <stdio.h>

static int nn_(BilinearSamplerThreeD_updateOutput)(lua_State *L)
{


  THTensor *inputImages = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *grids = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_checkudata(L, 4, torch_Tensor);
/* 2D version
  int batchsize = inputImages->size[0];
  int inputImages_height = inputImages->size[1];
  int inputImages_width = inputImages->size[2];
  int output_height = output->size[1];
  int output_width = output->size[2];
  int inputImages_channels = inputImages->size[3];
  

  int output_strideBatch = output->stride[0];
  int output_strideHeight = output->stride[1];
  int output_strideWidth = output->stride[2];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideHeight = inputImages->stride[1];
  int inputImages_strideWidth = inputImages->stride[2];

  int grids_strideBatch = grids->stride[0];
  int grids_strideHeight = grids->stride[1];
  int grids_strideWidth = grids->stride[2];
*/

// 3D version:
  int batchsize = inputImages->size[0];
  int inputImages_depth = inputImages->size[1];
  int inputImages_height = inputImages->size[2];
  int inputImages_width = inputImages->size[3];
  int output_depth = output->size[1];
  int output_height = output->size[2];
  int output_width = output->size[3];
  int inputImages_channels = inputImages->size[4];
  

  int output_strideBatch = output->stride[0];
  int output_strideDepth = output->stride[1];
  int output_strideHeight = output->stride[2];
  int output_strideWidth = output->stride[3];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideDepth = inputImages->stride[1];
  int inputImages_strideHeight = inputImages->stride[2];
  int inputImages_strideWidth = inputImages->stride[3];

  int grids_strideBatch = grids->stride[0];
  int grids_strideDepth = grids->stride[1];
  int grids_strideHeight = grids->stride[2];
  int grids_strideWidth = grids->stride[3];


  printf("input depth: %d,\ninput height: %d,\ninput width: %d,\noutput depth: %d,\noutput height: %d,\noutput width: %d,\ninput channel: %d,\noutput stride batch: %d,\noutput stride depth: %d,\noutput stride height: %d,\noutput stride width: %d \ninput stride batch:%d,\ninput stride depth:%d,\ninput stride height: %d,\ninput stride width:%d\n", inputImages_depth, inputImages_height, inputImages_width,output_depth, output_height,output_width, inputImages_channels,output_strideBatch,output_strideDepth, output_strideHeight,output_strideWidth,inputImages_strideBatch, inputImages_strideDepth, inputImages_strideHeight, inputImages_strideWidth);

  printf("grid stride batch: %d,\ngrid stride depth: %d,\ngrid stride height: %d\ngrid stride width:%d \n", grids_strideBatch, grids_strideDepth,grids_strideHeight, grids_strideWidth);


  real *inputImages_data, *output_data, *grids_data;
  inputImages_data = THTensor_(data)(inputImages);
  output_data = THTensor_(data)(output);
  grids_data = THTensor_(data)(grids);

  int b, yOut, xOut, zOut;
  int point_count = 0;
  for(b=0; b < batchsize; b++)
  {
    for(xOut=0; xOut < output_depth; xOut++)
    {
    	for(zOut=0; zOut < output_height; zOut++)
   	{
	      for(yOut=0; yOut < output_width; yOut++)
	      {
		//read the grid
		/*
	  		grid is serialize in c and stored in grids_data
			grids_data[0] = top_left_x, grids_data[1] = top_left_y, grids_data[2] top_left_z
		*/ 
		printf("the %dth point: \n", point_count);
		point_count++;
		real xf = grids_data[b*grids_strideBatch + xOut*grids_strideDepth + zOut*grids_strideHeight + yOut*grids_strideWidth];
		real yf = grids_data[b*grids_strideBatch + xOut*grids_strideDepth + zOut*grids_strideHeight + yOut*grids_strideWidth + 1];
		real zf = grids_data[b*grids_strideBatch + xOut*grids_strideDepth + zOut*grids_strideHeight + yOut*grids_strideWidth + 2];
		printf("xf: %f, zf: %f, yf: %f \n", xf, zf, yf);
		// get the weights for interpolation
		int yInTopLeft, xInTopLeft, zInTopLeft;
		real yWeightTopLeft, xWeightTopLeft, zWeightTopLeft;

		//convert the grid coordinate after transform back from normalise between [-1, 1], see AffineGridGeneratorBHWD for normalisation
		real xcoord = (xf + 1) * (inputImages_depth - 1) / 2;
		xInTopLeft = floor(xcoord);
		xWeightTopLeft = 1 - (xcoord - xInTopLeft);

		real ycoord = (yf + 1) * (inputImages_width - 1) / 2;
		yInTopLeft = floor(ycoord);
		yWeightTopLeft = 1 - (ycoord - yInTopLeft);

		real zcoord = (zf + 1) * (inputImages_height - 1) / 2;
		zInTopLeft = floor(zcoord);
		zWeightTopLeft = 1 - (zcoord - zInTopLeft);
		printf("real x:%f, real z: %f, real y: %f, \n", xcoord, zcoord, ycoord);

		
		// image coordinate
		const int outAddress = output_strideBatch * b + output_strideDepth * xOut + output_strideHeight * zOut +  output_strideWidth * yOut;
		// grid front top left coordinate
		const int inFrontTopLeftAddress = inputImages_strideBatch * b + inputImages_strideDepth * xInTopLeft + inputImages_strideHeight * zInTopLeft + inputImages_strideWidth * yInTopLeft;
		// get grid front top right coordinate from top left
		const int inFrontTopRightAddress = inFrontTopLeftAddress + inputImages_strideWidth;
		// get grid front bottom left from top left
		const int inFrontBottomLeftAddress = inFrontTopLeftAddress + inputImages_strideHeight;
		// get grid front bottom right from bottom left        
		const int inFrontBottomRightAddress = inFrontBottomLeftAddress + inputImages_strideWidth;
		
		//add
		// grid back top left coordinate
		const int inBackTopLeftAddress = inFrontTopLeftAddress + inputImages_strideDepth;
		// get grid back top right coordinate from top left
		const int inBackTopRightAddress = inBackTopLeftAddress + inputImages_strideWidth;
		// get grid back bottom left from top left
		const int inBackBottomLeftAddress = inBackTopLeftAddress + inputImages_strideHeight;
		// get grid back bottom right from bottom left        
		const int inBackBottomRightAddress = inBackBottomLeftAddress + inputImages_strideWidth;
		//end add

		real v=0;
		real inFrontTopLeft=0;
		real inFrontTopRight=0;
		real inFrontBottomLeft=0;
		real inFrontBottomRight=0;

		//add
		real inBackTopLeft=0;
		real inBackTopRight=0;
		real inBackBottomLeft=0;
		real inBackBottomRight=0;
		//end add

		// we are careful with the boundaries
		bool frontTopLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_depth-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_width-1  && zInTopLeft >= 0 && zInTopLeft <= inputImages_height-1;
		bool frontTopRightIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_depth-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_width-1 && zInTopLeft >= 0 && zInTopLeft <= inputImages_height-1;
		bool frontBottomLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_depth-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_width-1 && zInTopLeft+1 >= 0 && zInTopLeft+1 <= inputImages_height-1;
		bool frontBottomRightIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_depth-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_width-1 && zInTopLeft+1 >= 0 && zInTopLeft+1 <= inputImages_height-1;

		//add
		bool backTopLeftIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_depth-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_width-1 && zInTopLeft >= 0 && zInTopLeft <= inputImages_height-1;
		bool backTopRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_depth-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_width-1 && zInTopLeft >= 0 && zInTopLeft <= inputImages_height-1;
		bool backBottomLeftIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_depth-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_width-1 && zInTopLeft+1 >= 0 && zInTopLeft+1 <= inputImages_height-1;
		bool backBottomRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_depth-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_width-1 && zInTopLeft+1 >= 0 && zInTopLeft+1 <= inputImages_height-1;
		//end add

		int t;
		// interpolation happens here
		for(t=0; t<inputImages_channels; t++)
		{
		   if(frontTopLeftIsIn) inFrontTopLeft = inputImages_data[inFrontTopLeftAddress + t];
		   if(frontTopRightIsIn) inFrontTopRight = inputImages_data[inFrontTopRightAddress + t];
		   if(frontBottomLeftIsIn) inFrontBottomLeft = inputImages_data[inFrontBottomLeftAddress + t];
		   if(frontBottomRightIsIn) inFrontBottomRight = inputImages_data[inFrontBottomRightAddress + t];

		   //add
		   if(backTopLeftIsIn) inBackTopLeft = inputImages_data[inBackTopLeftAddress + t];
		   if(backTopRightIsIn) inBackTopRight = inputImages_data[inBackTopRightAddress + t];
		   if(backBottomLeftIsIn) inBackBottomLeft = inputImages_data[inBackBottomLeftAddress + t];
		   if(backBottomRightIsIn) inBackBottomRight = inputImages_data[inBackBottomRightAddress + t];
		   //end add

		   v = xWeightTopLeft * yWeightTopLeft * zWeightTopLeft * inFrontTopLeft
		     + xWeightTopLeft * (1 - yWeightTopLeft) * zWeightTopLeft * inFrontTopRight
		     + xWeightTopLeft * yWeightTopLeft * (1 - zWeightTopLeft) * inFrontBottomLeft
		     + xWeightTopLeft * (1 - yWeightTopLeft) * (1 - zWeightTopLeft) * inFrontBottomRight   
                     + (1 - xWeightTopLeft) * yWeightTopLeft * zWeightTopLeft * inBackTopLeft
                     + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * zWeightTopLeft * inBackTopRight 
		     + (1 - xWeightTopLeft) * yWeightTopLeft * (1 - zWeightTopLeft) * inBackBottomLeft
		     + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * (1 - zWeightTopLeft) * inBackBottomRight;
		   output_data[outAddress + t] = v;
		}

	      }
      	}
    }
  }
  return 1;
}

static int nn_(BilinearSamplerThreeD_updateGradInput)(lua_State *L)
{
  THTensor *inputImages = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *grids = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradInputImages = luaT_checkudata(L, 4, torch_Tensor);
  THTensor *gradGrids = luaT_checkudata(L, 5, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 6, torch_Tensor);

  bool onlyGrid=false;

  int batchsize = inputImages->size[0];
  int inputImages_height = inputImages->size[1];
  int inputImages_width = inputImages->size[2];
  int gradOutput_height = gradOutput->size[1];
  int gradOutput_width = gradOutput->size[2];
  int inputImages_channels = inputImages->size[3];

  int gradOutput_strideBatch = gradOutput->stride[0];
  int gradOutput_strideHeight = gradOutput->stride[1];
  int gradOutput_strideWidth = gradOutput->stride[2];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideHeight = inputImages->stride[1];
  int inputImages_strideWidth = inputImages->stride[2];

  int gradInputImages_strideBatch = gradInputImages->stride[0];
  int gradInputImages_strideHeight = gradInputImages->stride[1];
  int gradInputImages_strideWidth = gradInputImages->stride[2];

  int grids_strideBatch = grids->stride[0];
  int grids_strideHeight = grids->stride[1];
  int grids_strideWidth = grids->stride[2];

  int gradGrids_strideBatch = gradGrids->stride[0];
  int gradGrids_strideHeight = gradGrids->stride[1];
  int gradGrids_strideWidth = gradGrids->stride[2];

  real *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = THTensor_(data)(inputImages);
  gradOutput_data = THTensor_(data)(gradOutput);
  grids_data = THTensor_(data)(grids);
  gradGrids_data = THTensor_(data)(gradGrids);
  gradInputImages_data = THTensor_(data)(gradInputImages);

  int b, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    for(yOut=0; yOut < gradOutput_height; yOut++)
    {
      for(xOut=0; xOut < gradOutput_width; xOut++)
      {
        //read the grid
        real yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth];
        real xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + 1];

        // get the weights for interpolation
        int yInTopLeft, xInTopLeft;
        real yWeightTopLeft, xWeightTopLeft;
 
        real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
        xInTopLeft = floor(xcoord);
        xWeightTopLeft = 1 - (xcoord - xInTopLeft);

        real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
        yInTopLeft = floor(ycoord);
        yWeightTopLeft = 1 - (ycoord - yInTopLeft);

        
        const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
        const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
        const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
        const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

        const int gradInputImagesTopLeftAddress = gradInputImages_strideBatch * b + gradInputImages_strideHeight * yInTopLeft + gradInputImages_strideWidth * xInTopLeft;
        const int gradInputImagesTopRightAddress = gradInputImagesTopLeftAddress + gradInputImages_strideWidth;
        const int gradInputImagesBottomLeftAddress = gradInputImagesTopLeftAddress + gradInputImages_strideHeight;
        const int gradInputImagesBottomRightAddress = gradInputImagesBottomLeftAddress + gradInputImages_strideWidth;

        const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight * yOut + gradOutput_strideWidth * xOut;

        real topLeftDotProduct = 0;
        real topRightDotProduct = 0;
        real bottomLeftDotProduct = 0;
        real bottomRightDotProduct = 0;

        real v=0;
        real inTopLeft=0;
        real inTopRight=0;
        real inBottomLeft=0;
        real inBottomRight=0;

        // we are careful with the boundaries
        bool topLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool topRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool bottomLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;
        bool bottomRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;

        int t;

        for(t=0; t<inputImages_channels; t++)
        {
           real gradOutValue = gradOutput_data[gradOutputAddress + t];
           if(topLeftIsIn)
           {
              real inTopLeft = inputImages_data[inTopLeftAddress + t];
              topLeftDotProduct += inTopLeft * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopLeftAddress + t] += xWeightTopLeft * yWeightTopLeft * gradOutValue;
           }

           if(topRightIsIn)
           {
              real inTopRight = inputImages_data[inTopRightAddress + t];
              topRightDotProduct += inTopRight * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopRightAddress + t] += (1 - xWeightTopLeft) * yWeightTopLeft * gradOutValue;
           }

           if(bottomLeftIsIn)
           {
              real inBottomLeft = inputImages_data[inBottomLeftAddress + t];
              bottomLeftDotProduct += inBottomLeft * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomLeftAddress + t] += xWeightTopLeft * (1 - yWeightTopLeft) * gradOutValue;
           }

           if(bottomRightIsIn)
           {
              real inBottomRight = inputImages_data[inBottomRightAddress + t];
              bottomRightDotProduct += inBottomRight * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomRightAddress + t] += (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * gradOutValue;
           }
        }

        yf = - xWeightTopLeft * topLeftDotProduct + xWeightTopLeft * bottomLeftDotProduct - (1-xWeightTopLeft) * topRightDotProduct + (1-xWeightTopLeft) * bottomRightDotProduct;
        xf = - yWeightTopLeft * topLeftDotProduct + yWeightTopLeft * topRightDotProduct - (1-yWeightTopLeft) * bottomLeftDotProduct + (1-yWeightTopLeft) * bottomRightDotProduct;

        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth] = yf * (inputImages_height-1) / 2;
        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + 1] = xf * (inputImages_width-1) / 2;

      }
    }
  }

  return 1;
}

static const struct luaL_Reg nn_(BilinearSamplerThreeD__) [] = {
  {"BilinearSamplerThreeD_updateOutput", nn_(BilinearSamplerThreeD_updateOutput)},
  {"BilinearSamplerThreeD_updateGradInput", nn_(BilinearSamplerThreeD_updateGradInput)},
  {NULL, NULL}
};

static void nn_(BilinearSamplerThreeD_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(BilinearSamplerThreeD__), "nn");
  lua_pop(L,1);
}

#endif
