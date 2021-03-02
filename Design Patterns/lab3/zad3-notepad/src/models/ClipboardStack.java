package models;

import interfaces.ClipboardObserver;

import java.awt.datatransfer.Clipboard;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

public class ClipboardStack {
    private Stack<String> stack = new Stack<>();
    private List<ClipboardObserver> observers = new LinkedList<>();

    public void push(String text){
        stack.push(text);
    }

    public String pop(){
        return stack.pop();
    }

    public String peek(){
        return stack.peek();
    }

    public void clear(){
        stack.clear();
    }

    public boolean isEmpty(){
        return stack.isEmpty();
    }

    public void addObserver(ClipboardObserver obs){
        observers.add(obs);
    }

    public void removeObserver(ClipboardObserver obs){
        observers.remove(obs);
    }

    public void notifyObservers(){
        for(ClipboardObserver o : observers){
            o.updateClipboard();
        }
    }
}
