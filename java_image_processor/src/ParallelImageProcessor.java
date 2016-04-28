import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by zuhair on 4/27/16.
 */
public class ParallelImageProcessor {

    public static Index BLOCK_DIMENSION = new Index(960, 1080, 1);

    public static void main(String[] args){
        ArrayList<Long> times = new ArrayList<>();
        try {
            File folder = new File("/Users/zuhair/prog/ee360p/pictures");
            File[] listOfFiles = folder.listFiles();
            //File[] listOfFiles = new File[] { new File("/Users/zuhair/prog/ee360p/pictures/car.jpg")};
            ExecutorService executorService = Executors.newCachedThreadPool();
            int count = 0;
            for(File imageFile : listOfFiles) {
                if(!imageFile.getName().contains("jpg")) continue;
                System.out.print(String.format("%03d %s    \r", count, imageFile.getName()));
                System.out.flush();

                BufferedImage image = ImageIO.read(imageFile);
                int[] a = image.getRGB(0, 0, image.getWidth(), image.getHeight(), null, 0, image.getWidth());

                BLOCK_DIMENSION = new Index(image.getWidth()/4, image.getHeight()/2, 1);

                Index gridDimensions = new Index(
                        ((int) Math.ceil(image.getWidth() / BLOCK_DIMENSION.x)),
                        ((int) Math.ceil(image.getHeight() / BLOCK_DIMENSION.y)),
                        1
                );

                ArrayList<PixelProcessor> pixelProcessors = new ArrayList<>();
                //ArrayList<ColorProcessor> pixelProcessors = new ArrayList<>();

                int[] output = new int[a.length];

//                int[] red = new int[a.length];
//                int[] green = new int[a.length];
//                int[] blue = new int[a.length];
                for (int r = 0; r < gridDimensions.y; r++) {
                    for (int c = 0; c < gridDimensions.x; c++) {
                        Index blockIndex = new Index(c, r, 0);
            //            pixelProcessors.add(new EmbossenFilter(red, green, blue, a, image.getWidth(), image.getHeight(), blockIndex));
                        pixelProcessors.add(new ObamaifyFilter(output, a, image.getWidth(), image.getHeight(), blockIndex));
                    }
                }


                long start = System.currentTimeMillis();
                //pixelProcessors.forEach(ColorProcessor::call);
                executorService.invokeAll(pixelProcessors);
                long end = System.currentTimeMillis();
                times.add(end-start);

//                for(int i=0; i < output.length; i++){
//                    output[i] = ((red[i]) << 16) | ((green[i]) << 8) | (blue[i]);
//                }

                BufferedImage outputImage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_RGB);
                outputImage.setRGB(0, 0, image.getWidth(), image.getHeight(), output, 0, image.getWidth());

                ImageIO.write(outputImage, "jpg", new File("/Users/zuhair/prog/ee360p/output/"+imageFile.getName()));
                count++;

                System.gc();
            }
            double sum = 0;
            for(long l : times){
                sum += l;
            }
            System.out.println(sum/times.size() + " milliseconds");

            for(long l : times){
                System.out.println(l);
            }


        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }
}