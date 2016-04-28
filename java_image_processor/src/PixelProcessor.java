import java.util.concurrent.Callable;

/**
 * Created by zuhair on 4/27/16.
 */
public abstract class PixelProcessor implements Callable<Void> {

    public static final int R = 0;
    public static final int G = 1;
    public static final int B = 2;

    int[] output;
    int[] input;
    int width, height;

    Index blockIndex;

    public PixelProcessor(int[] dest, int[] a, int width, int height, Index blockIndex){
        this.input = a;
        this.output = dest;

        this.width = width;
        this.height = height;

        this.blockIndex = blockIndex;
    }

    public Void call() {
        for(int row = 0; row < ParallelImageProcessor.BLOCK_DIMENSION.y; row++){
            for(int col = 0; col < ParallelImageProcessor.BLOCK_DIMENSION.x; col++){
                Index threadIndex = new Index(row, col, 0);
                processPixel(threadIndex);
            }
        }
        return null;
    }

    public abstract void processPixel(Index threadIndex);
}
