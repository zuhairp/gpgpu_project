/**
 * Created by zuhair on 4/27/16.
 */
public class Index {
    public int x, y, z;
    public Index(int x, int y, int z){
        this.x = x;
        this.y = y;
        this.z = z;
    }

    @Override
    public String toString() {
        return String.format("(%d, %d, %d) ", x, y, z);
    }
}
