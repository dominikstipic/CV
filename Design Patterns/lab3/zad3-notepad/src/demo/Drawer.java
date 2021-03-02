package demo;

import javax.swing.*;
import java.awt.*;

public class Drawer extends JComponent {


    /*@Override
    public Dimension getPreferredSize() {
        return new Dimension(500,500);
    }*/

    @Override
    protected void paintComponent(Graphics g) {
        Graphics2D g2 = (Graphics2D) g;
        g2.setColor(Color.RED);

        // točka veličine 1 pixel
        g2.setStroke(new BasicStroke(1));
        // vodoravna linija
        g2.drawLine(100,300,200,300);

        // okomita linija
        g2.drawLine(300,300,300,100);

        g.fillRect(10,10,100,100);

        g2.setFont(new Font("TimesRoman", Font.PLAIN, 13));
        g2.setColor(Color.BLACK);
        g2.drawString("Hello world",10,20);
        g2.drawString("Hello world",10,33);
    }

}
