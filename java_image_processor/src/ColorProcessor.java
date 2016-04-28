import java.util.concurrent.Callable;

/**
 * Created by zuhair on 4/27/16.
 */
public abstract class ColorProcessor implements Callable<Void>{
    public static final int R = 0;
    public static final int G = 1;
    public static final int B = 2;

    int[] redOutput, blueOutput, greenOutput;
    int[] input;
    int width, height;

    Index blockIndex;

    public ColorProcessor(int[] redOut, int[] greenOut, int[] blueOut, int[] a, int width, int height, Index blockIndex){
        this.input = a;

        this.redOutput = redOut;
        this.greenOutput = greenOut;
        this.blueOutput = blueOut;

        this.width = width;
        this.height = height;

        this.blockIndex = blockIndex;
    }

    public Void call() {
        for(int row = 0; row < ParallelImageProcessor.BLOCK_DIMENSION.y; row++){
            for(int col = 0; col < ParallelImageProcessor.BLOCK_DIMENSION.x; col++){
                for(int color = 0; color < 3; color++) {
                    Index threadIndex = new Index(row, col, color);
                    processColor(threadIndex);
                }
            }
        }
        return null;
    }

    public abstract void processColor(Index threadIndex);
}
