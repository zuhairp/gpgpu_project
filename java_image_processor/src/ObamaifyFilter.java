/**
 * Created by zuhair on 4/27/16.
 */
public class ObamaifyFilter extends PixelProcessor {

    public ObamaifyFilter(int[] dest, int[] a, int width, int height, Index blockIndex) {
        super(dest, a, width, height, blockIndex);
    }

    @Override
    public void processPixel(Index threadIndex) {
        int rowIndex = ParallelImageProcessor.BLOCK_DIMENSION.y*blockIndex.y + threadIndex.y;
        int colIndex = ParallelImageProcessor.BLOCK_DIMENSION.x*blockIndex.x + threadIndex.x;

        int index = width*rowIndex + colIndex;

        if(rowIndex >= height) return;
        if(colIndex >= width) return;

        int[] red = new int[] { 0xFC, 0x44, 0x4D };
        int[] white = new int[] { 255, 255, 255 };
        int[] blue = new int[] { 0x46, 0x27, 0xF5 };

        int rgb = input[index];
        int r = (rgb >> 16) & 0xFF;
        int g = (rgb >> 8) & 0xFF;
        int b = (rgb >> 0) & 0xFF;

        int[] val = new int[] {r, g, b};

        int redDistance = (red[0]-val[0])*(red[0]-val[0]) + (red[1]-val[1])*(red[1]-val[1]) + (red[2]-val[2])*(red[2]-val[2]);
        int blueDistance = (blue[0]-val[0])*(blue[0]-val[0]) + (blue[1]-val[1])*(blue[1]-val[1]) + (blue[2]-val[2])*(blue[2]-val[2]);
        int whiteDistance = (white[0]-val[0])*(white[0]-val[0]) + (white[1]-val[1])*(white[1]-val[1]) + (white[2]-val[2])*(white[2]-val[2]);

        int[] result = new int[] {0, 0, 0};
        if(redDistance <= blueDistance && redDistance <= whiteDistance) result = red;
        if(blueDistance <= redDistance && blueDistance <= whiteDistance) result = blue;
        if(whiteDistance <= redDistance && whiteDistance <= blueDistance) result = white;

        output[index] = (result[0]<<16) | (result[1] << 8) | (result[2]);
    }
}
