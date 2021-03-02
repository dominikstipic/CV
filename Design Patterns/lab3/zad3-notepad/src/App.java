import models.Location;
import textEditor.TextEditor;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class App extends JFrame{
    private TextEditor editor;
    private JMenuBar menuBar = new JMenuBar();

    private int height = 600;
    private int width = 600;

    public App() {
        setTitle("Text editor");
        setSize(width,height);
        setLocation(100,100);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        initGui();
    }

    private void initGui(){
        editor = new TextEditor(height,width);
        getContentPane().setLayout(new BorderLayout());
        getContentPane().add(editor,BorderLayout.CENTER);
        getContentPane().add(menuBar, BorderLayout.NORTH);


        JMenu file = new JMenu("File");
        JMenu edit = new JMenu("Edit");
        JMenu move = new JMenu("Move");
        menuBar.add(file);
        menuBar.add(edit);
        menuBar.add(move);

        JMenuItem open = new JMenuItem(new AbstractAction("open") {
            @Override
            public void actionPerformed(ActionEvent e) {
                final JFileChooser fc = new JFileChooser();
                int val = fc.showOpenDialog(App.this);
                if(val == JFileChooser.APPROVE_OPTION){
                    File file = fc.getSelectedFile();
                    try {
                        List<String> lines = Files.readAllLines(file.toPath());
                        lines.stream().forEach(s -> {
                            editor.getModel().insert(s);
                            editor.getModel().newLine();
                        });
                    } catch (IOException e1) {
                        e1.printStackTrace();
                    }

                }
            }
        });

        JMenuItem save = new JMenuItem (new AbstractAction("save") {
            @Override
            public void actionPerformed (ActionEvent e) {
                final JFileChooser fc = new JFileChooser();
                int val = fc.showSaveDialog(App.this);
                if(val == JFileChooser.APPROVE_OPTION){
                    try{
                        File file = fc.getSelectedFile();
                        Path path = file.toPath();
                        if(!Files.exists(file.toPath())){
                            path = Files.createFile(path);
                        }

                        Iterator<Map.Entry<Integer, List<Character>>> it = editor.getModel().allLines();
                        StringBuilder sb = new StringBuilder();
                        while(it.hasNext()){
                            Map.Entry<Integer, List<Character>> entry = it.next();
                            for(Character c:entry.getValue()){
                                sb.append(c);
                            }
                            sb.append("\n");
                        }
                        Files.writeString(path,sb.toString());
                    }catch (IOException exc){

                    }

                }
            }
        });

        JMenuItem exit = new JMenuItem(new AbstractAction("exit") {
            @Override
            public void actionPerformed(ActionEvent e) {
                System.exit(0);
            }
        });

        file.add(open);
        file.add(save);
        file.add(exit);

        JMenuItem undo = new JMenuItem("undo");
        JMenuItem redo = new JMenuItem("redo");
        JMenuItem cut = new JMenuItem("cut");
        JMenuItem copy = new JMenuItem("copy");
        JMenuItem paste = new JMenuItem("paste");

        edit.add(undo);
        edit.add(redo);
        edit.add(cut);
        edit.add(copy);
        edit.add(paste);

        JMenuItem start = new JMenuItem(new AbstractAction("Cursor to document start") {
            @Override
            public void actionPerformed(ActionEvent e) {
                editor.getModel().setCursorLocation(new Location(0,0));
                repaint();

            }
        });
        JMenuItem end = new JMenuItem("Cursor to document end");

        move.add(start);
        move.add(end);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new App().setVisible(true));
    }
}

