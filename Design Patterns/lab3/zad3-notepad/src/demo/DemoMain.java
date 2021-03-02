package demo;

import javax.swing.*;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

public class DemoMain extends JFrame{
    private Drawer drawer = new Drawer();
    private KeyListener keyListener ;

    private int height = 600;
    private int width = 600;

    public DemoMain() {
        setTitle("Demo prozor");
        setSize(width,height);
        setLocation(100,100);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);


        initGui();
    }

    private void initGui(){

        keyListener = new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                int code = e.getKeyCode();
                if(code == 10){
                    System.exit(0);
                }
            }
        };

        getContentPane().add(drawer);
        addKeyListener(keyListener);
    }



    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new DemoMain().setVisible(true));
    }
}
