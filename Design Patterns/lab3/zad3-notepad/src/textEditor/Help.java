package textEditor;

import java.awt.*;
import java.util.Iterator;

public class Help {

    public static void main(String[] args) {
        Font font = new Font("Helvetica",Font.PLAIN,12);
        Canvas c = new Canvas();
        FontMetrics fm = c.getFontMetrics(font);
    }

    public static void printIterator(Iterator it){
        while (it.hasNext()){
            System.out.print(it.next());
        }
        System.out.println();
    }
}
