package textEditor;

import interfaces.CursorObserver;
import interfaces.TextObserver;
import models.Location;
import models.LocationRange;

import java.awt.*;
import java.util.*;
import java.util.List;

public class TextEditorModel {
    private Map<Integer,List<Character>> linesMap;
    private LocationRange selectionRange;

    private Location cursorLocation;

    public Location getCursorLocation() {
        return new Location(cursorLocation.getX(),cursorLocation.getY());
    }

    public void setCursorLocation(Location cursorLocation) {
        this.cursorLocation = cursorLocation;
    }

    private List<CursorObserver> cursorObserver = new LinkedList<>();
    private List<TextObserver> textObserver = new LinkedList<>();

    private Font font;

    public TextEditorModel(int width, int height, Font font){

        this.font = font;

        // 1 row and 1 column -> cursor is in front of letter in (row = 0,column = 0)
        cursorLocation = new Location(0,0);
        selectionRange = null;
        linesMap       = new HashMap<>();
        linesMap.put(0,new LinkedList<>());
    }


    public Iterator<Map.Entry<Integer, List<Character>>> allLines(){
        List<Map.Entry<Integer, List<Character>>> lines = new LinkedList<>(linesMap.entrySet());
        lines.sort((e1,e2) -> e1.getKey().compareTo(e2.getKey()));
        return lines.iterator();
    }


    public Iterator<Character> linesRange(Location start, Location end) {
       if(end.equals(start)){
           throw new IllegalArgumentException("start cannot be same as end");
       }
       int startRow = start.getY();
       int endRow   = end.getY();

       int startColumn = start.getX();
       int endColumn   = end.getX();

       Set<Integer> keys = linesMap.keySet();
       if(!keys.contains(startRow) || !keys.contains(endRow)){
           throw new IllegalArgumentException("Selected field which doesn't exist");
       }


       if(endRow == startRow){
           List<Character> list = linesMap.get(startRow);
           List<Character> acc = new LinkedList<>();

           int beginIter = endColumn > startColumn ? startColumn : endColumn;
           int endIter   = endColumn == beginIter ? startColumn : endColumn;

           for(int i = beginIter ; i < endIter; ++i){
               acc.add(list.get(i));
           }

           return acc.iterator();
       }

        Iterator<Map.Entry<Integer, List<Character>>> it = allLines();
        List<Character> accumulator = new LinkedList<>();
        while(it.hasNext()){
            Map.Entry<Integer, List<Character>> entry = it.next();
            List<Character> currentList = entry.getValue();
            int row = entry.getKey();

            if(row > startRow && row < endRow){
                //red koji je između
                List<Character> list = linesMap.get(row);
                accumulator.addAll(list);
            }
            else if(row == startRow ){
                //spremi sve od početka do kraja
                for(int i = startColumn; i < currentList.size(); ++i){
                    accumulator.add(currentList.get(i));
                }
            }
            else if(row == endRow){
                //oznaka u istom redku
                for(int i = 0; i < endColumn; ++i){
                    accumulator.add(currentList.get(i));
                }
            }
        }
    return accumulator.iterator();
    }

    public Iterator<Character> linesRange(LocationRange range){
        return linesRange(range.getStart(),range.getEnd());
    }

////////////////CURSOR OPERATION //////////////
    public void deleteBefore(){
        int row = cursorLocation.getY();
        int column = cursorLocation.getX();
        if(column == 0){
            moveCursorUp();
            return;
        }

        List<Character> line = linesMap.get(row);
        line.remove(column-1);
        moveCursorLeft();
    }

    public void deleteAfter(){
        int row = cursorLocation.getY();
        int column = cursorLocation.getX();
        List<Character> line = linesMap.get(row);
        if(column == line.size()){
            return;
        }

        line.remove(column);
        notifyObservers();
    }

    public void deleteRange(LocationRange range){
        Location start = range.getStart();
        Location end   = range.getEnd();

        if(start.getY() == end.getY()){
            List<Character> list = linesMap.get(start.getY());
            int num = Math.abs(end.getX() - start.getX());

            boolean value = end.getX() > start.getX();
            Location head = value ? start : end;

            while(num != 0){
                list.remove(head.getX());
                --num;
                if(value) moveCursorLeft();
            }
            linesMap.put(start.getY(),list);
            return;
        }
    }

    public LocationRange getSelectionRange(){
        return selectionRange;
    }

    public void setSelectionRange(LocationRange range){
        if(range == null){
            selectionRange = null;
        }
        else if (selectionRange != null){
            selectionRange.setEnd(range.getEnd());
            return;
        }
        else {
            selectionRange = range;
        }
    }

    public void newLine(){
        int row    = cursorLocation.getY();
        int column = cursorLocation.getX();

        List<Character> newLine = new LinkedList<>();
        LocationRange range = new LocationRange(cursorLocation, new Location(linesMap.get(row).size(),row));
        Iterator<Character> it = linesRange(range);

        while(it.hasNext()){
            newLine.add(it.next());
        }
        linesMap.put(row+1,newLine);

        deleteRange(range);
        moveCursorDown();


    }


    /**
     * Get character font width
     * @param chr
     * @return font width
     */
    int getFontWidth(char chr){
        Canvas c = new Canvas();
        FontMetrics fm = c.getFontMetrics(font);
        return fm.charWidth(chr);
    }

    /**
     * Insert character into a current cursor line
     * @param c
     */
    public void insert(char c){
        int row = cursorLocation.getY();

        List<Character> line = linesMap.get(row);
        if(line == null){
            line = new LinkedList<>();
        }

        line.add(c);
        linesMap.put(row,line);
        moveCursorRight();
    }


    public void insert(String text){
        int row = cursorLocation.getY();
        List<Character> chars = new LinkedList<>();

        for(int i = 0; i < text.length(); ++i){
            char c = text.charAt(i);
            insert(c);
        }
    }

////////////////CURSOR OPERATION //////////////

    public void addCursorObserver(CursorObserver observer){
        cursorObserver.add(observer);
    }

    public void removeCursorObserver(CursorObserver observer){
        cursorObserver.remove(observer);
    }

    public void addTextObserver(TextObserver observer){
        textObserver.add(observer);
    }

    public void removeTextObserver(TextObserver observer){
        textObserver.remove(observer);
    }

    public void notifyObservers(){
        for(CursorObserver obs : cursorObserver){
            obs.updateCursorLocation(cursorLocation);
        }

        for(TextObserver obs : textObserver){
            obs.updateText();
        }
    }

/////////////OBSERVERS/////////////////////////


    /**
     * Returns real offset between :
     *  * Starting selection point and ending selection point
     * @return
     */
    public int getSelectionOffset(){
        int sum = 0;

        Iterator <Character> it = linesRange(selectionRange.getStart(),selectionRange.getEnd());
        while(it.hasNext()){
            Character c = it.next();
            sum += getFontWidth(c);
        }
        return sum;
    }

    public int getOffsetFromStart(Location point){
        int row = point.getY();
        int column = point.getX();

        List<Character> line = linesMap.get(row);
        if(line == null){
            return 0;
        }
        int sum = 0;
        for(int i = 0; i < column; ++i){
            Character c = line.get(i);
            sum += getFontWidth(c);
        }
        return sum;
    }

    public boolean isCursorInLines(){
        int row    = cursorLocation.getY();
        int column = cursorLocation.getX();

        // don't allow movement outside the text line
        List<Character> line = linesMap.get(row);
        return (column >= 0 && line != null && column < line.size());
    }


    public void moveCursorLeft(){
        int column = cursorLocation.getX();
        if(column > 0){
            cursorLocation.setX(cursorLocation.getX() - 1);
            notifyObservers();
        }
    }

    public void moveCursorRight(){
        int column = cursorLocation.getX();
        if(isCursorInLines()){
            cursorLocation.setX(cursorLocation.getX() + 1);
            notifyObservers();
        }
    }

    public void moveCursorUp(){
        int row = cursorLocation.getY();
        if(row > 0){
            cursorLocation.setY(row - 1);
            cursorLocation.setX(linesMap.get(row-1).size());
            notifyObservers();
        }
    }

    public void moveCursorDown(){
        int row = cursorLocation.getY();
        if(linesMap.get(row+1) != null){
            cursorLocation.setY(row + 1);
            cursorLocation.setX(linesMap.get(row+1).size());
            notifyObservers();
        }
    }

}
