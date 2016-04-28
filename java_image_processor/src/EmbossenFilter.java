/**
 * Created by zuhair on 4/27/16.
 */
public class EmbossenFilter extends ColorProcessor {

    public EmbossenFilter(int[] redOut, int[] greenOut, int[] blueOut, int[] a, int width, int height, Index blockIndex) {
        super(redOut, greenOut, blueOut, a, width, height, blockIndex);
    }

    public int getIndex(int rowIndex, int colIndex){
        if(rowIndex >= height) rowIndex = height-1;
        if(colIndex >= width) colIndex = width-1;

        if(rowIndex < 0) rowIndex = 0;
        if(colIndex < 0) colIndex = 0;;

        return rowIndex*width + colIndex;
    }

    @Override
    public void processColor(Index threadIndex) {
        int rowIndex = ParallelImageProcessor.BLOCK_DIMENSION.y*blockIndex.y + threadIndex.y;
        int colIndex = ParallelImageProcessor.BLOCK_DIMENSION.x*blockIndex.x + threadIndex.x;

        int index = getIndex(rowIndex, colIndex);

        if(rowIndex >= height) return;
        if(colIndex >= width) return;

        int convolution_matrix[] = {
            -4, -2,  5,
            -1,  1,  2,
            -5,  2,  3,
        };

        int result = 0;

        int r, c;
        int count = 0;
        for(r=-1; r <= 1; r++){
            for(c=-1; c <= 1; c++){
                int multiplier = convolution_matrix[count];
                int neighbor_index = getIndex(rowIndex+r, colIndex+c);
                int neighbor = input[neighbor_index];

                switch(threadIndex.z){
                    case R: neighbor = (neighbor >> 16) & 0xFF; break;
                    case G: neighbor = (neighbor >>  8) & 0xFF; break;
                    case B: neighbor = (neighbor >>  0) & 0xFF; break;
                }

                count += 1;
                result += multiplier * neighbor;
            }
        }

        if(result < 0) result = 0;
        if(result > 255) result = 255;

        switch(threadIndex.z){
            case R: redOutput[index] = result; break;
            case G: greenOutput[index] = result; break;
            case B: blueOutput[index] = result; break;
        }

    }
}
