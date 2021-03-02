package textEditor;

import interfaces.ClipboardObserver;
import interfaces.State;
import models.ClipboardStack;
import models.Location;
import models.LocationRange;
import models.TextState;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.InputEvent;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class TextEditor extends JComponent {
    protected TextEditorModel model;
    private Font font = new Font("serif", Font.PLAIN, 20);
    private ClipboardStack editorClipboard = new ClipboardStack();

    private State state = new TextState();

    public TextEditor(int width, int height) {
        model = new TextEditorModel(width,height,font);
        addKeyListeners();
        addActions();
        model.addCursorObserver((loc) -> repaint());
        model.addTextObserver(() -> repaint());

        editorClipboard.addObserver(new ClipboardObserver() {
            @Override
            public void updateClipboard() {

            }
        });
    }

    private void addActions(){
        getActionMap().put("shiftLeft", new AbstractAction() {
            public void actionPerformed(ActionEvent e){
                Location start = model.getCursorLocation();
                model.moveCursorLeft();
                Location end = model.getCursorLocation();
                LocationRange range = new LocationRange(start,end);
                model.setSelectionRange(range);
                repaint();
            }});

        getActionMap().put("shiftRight", new AbstractAction() {
            public void actionPerformed(ActionEvent e){
                Location start = model.getCursorLocation();
                model.moveCursorRight();
                Location end = model.getCursorLocation();
                LocationRange range = new LocationRange(start,end);
                model.setSelectionRange(range);
                repaint();
            }});

        getActionMap().put("shiftUp", new AbstractAction() {
            public void actionPerformed(ActionEvent e){
            }});
        getActionMap().put("shiftDown", new AbstractAction() {
            public void actionPerformed(ActionEvent e){

            }});

        getActionMap().put("up", new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if(model.getSelectionRange() != null){
                    model.setSelectionRange(null);
                    repaint();
                }
                else{
                    model.moveCursorUp();
                }
            }
        });
        getActionMap().put("down", new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if(model.getSelectionRange() != null){
                    model.setSelectionRange(null);
                    repaint();
                }
                else{
                    model.moveCursorDown();
                }
            }
        });
        getActionMap().put("left", new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if(model.getSelectionRange() != null){
                    model.setSelectionRange(null);
                    repaint();
                }
                else{
                    model.moveCursorLeft();
                }
            }
        });
        getActionMap().put("right", new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if(model.getSelectionRange() != null){
                    model.setSelectionRange(null);
                    repaint();
                }
                else{
                    model.moveCursorRight();
                }
            }
        });

        getActionMap().put("enter", new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                model.setSelectionRange(null);
                model.newLine();
                repaint();
            }
        });
        getActionMap().put("backspace", new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if(model.getSelectionRange() != null){
                    model.deleteRange(model.getSelectionRange());
                    model.setSelectionRange(null);
                    repaint();
                }
                else{
                    model.deleteBefore();
                }
            }
        });
        getActionMap().put("delete", new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if(model.getSelectionRange() != null){
                    model.deleteRange(model.getSelectionRange());
                    model.setSelectionRange(null);
                    repaint();
                }
                else{
                    model.deleteAfter();
                }
            }
        });

        getActionMap().put("copy", new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                LocationRange range = model.getSelectionRange();
                if(range != null){
                    Iterator<Character> it =  model.linesRange(range);
                    StringBuilder sb = new StringBuilder();
                    while(it.hasNext()){
                        Character c = it.next();
                        sb.append(c);
                    }
                    editorClipboard.push(sb.toString());
                }
            }
        });

        getActionMap().put("paste", new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if(!editorClipboard.isEmpty()){
                    String text = editorClipboard.peek();
                    model.insert(text);
                    repaint();
                }
            }
        });

        getActionMap().put("cut", new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                getActionMap().get("copy").actionPerformed(null);
                LocationRange range = model.getSelectionRange();
                model.deleteRange(range);
                model.setSelectionRange(null);
                repaint();
            }
        });

    }

    private void addKeyListeners(){
        getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_LEFT, InputEvent.SHIFT_DOWN_MASK), "shiftLeft");
        getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_RIGHT, InputEvent.SHIFT_DOWN_MASK), "shiftRight");
        getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_UP, InputEvent.SHIFT_DOWN_MASK), "shiftUp");
        getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_DOWN, InputEvent.SHIFT_DOWN_MASK), "shiftDown");

        getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_C, InputEvent.CTRL_DOWN_MASK), "copy");
        getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_X, InputEvent.CTRL_DOWN_MASK), "cut");
        getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_V, InputEvent.CTRL_DOWN_MASK), "paste");


        getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_ENTER, 0), "enter");
        getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_BACK_SPACE,0), "backspace");
        getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_DELETE, 0), "delete");

        getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_UP, 0), "up");
        getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_DOWN, 0), "down");
        getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_LEFT, 0), "left");
        getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_RIGHT, 0), "right");


        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                state.action(TextEditor.this, e);
            }
        });
    }


    @Override
    protected void paintComponent(Graphics g) {
        Graphics2D g2 = (Graphics2D) g;
        g2.setBackground(Color.WHITE);
        System.out.println(model.getCursorLocation());
        g2.setFont(font);

        drawCursor(g2);
        drawSelection(g);
        drawLines(g);
    }

    private void drawSelection(Graphics g){
        if(model.getSelectionRange() == null){
            return;
        }

        Location startLocation = model.getSelectionRange().getStart();
        Location endLocation   = model.getSelectionRange().getEnd();
        Location beginning     = new Location(startLocation.getX(),startLocation.getY());

        if(startLocation.getX() > endLocation.getX()){
            beginning = new Location(endLocation.getX(),endLocation.getY());
        }

        int rowOffset = font.getSize();
        int columnOffset = font.getSize();

        int width  = model.getSelectionOffset();
        int height = font.getSize();

        int startOffset = model.getOffsetFromStart(beginning);
        g.setColor(Color.orange);
        g.fillRect(rowOffset + startOffset , columnOffset/2 + rowOffset*startLocation.getY(),width,height);
    }

    private void drawCursor(Graphics2D g){
        g.setColor(Color.BLACK);
        int cursorRowOffsetFromStart = model.getOffsetFromStart(model.getCursorLocation());
        int delta = font.getSize();
        int rowOffset = font.getSize();
        int columnOffset = font.getSize();

        int row = model.getCursorLocation().getY() + 1;
        g.drawLine(rowOffset + cursorRowOffsetFromStart,(row*columnOffset+delta/2),rowOffset + cursorRowOffsetFromStart,(row*columnOffset-delta/2));
    }

    public void drawLines(Graphics g){
        g.setColor(Color.BLACK);
        Iterator<Map.Entry<Integer, List<Character>>> it = model.allLines();
        while(it.hasNext()){
            Map.Entry<Integer, List<Character>> entry = it.next();
            int row = entry.getKey();
            String line = entry.getValue().stream().map(c -> Character.toString(c)).collect(Collectors.joining());

            int offset = font.getSize();
            g.drawString(line,offset,((row +1 )*offset + offset/2));
        }
    }

    public void addCharacter(char letter){
        if(String.valueOf(letter).equals(System.getProperty("line.separator"))){
            model.newLine();
        }
        model.insert(letter);
    }

    public TextEditorModel getModel() {
        return model;
    }

    public void setState(State state) {
        this.state = state;
    }
}
