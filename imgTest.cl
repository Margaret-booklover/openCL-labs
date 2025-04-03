__kernel void imgTest(
	__read_only  image2d_t img1,
	__write_only image2d_t img2
) {
							//Natural coordinates		Clamp to zeros		Don't interpolate
	const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	uint4 val = read_imageui(img1, smp, coord);
	val.y = 255-val.y;val.z = 255-val.z;val.x = 255-val.x;
	//val.y = val.z = val.x =  0.36 * val.x + 0.53 * val.y + 0.11 * val.z;
	write_imageui(img2, coord, val); 
};