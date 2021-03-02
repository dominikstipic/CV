package models;

import interfaces.State;
import textEditor.TextEditor;

import java.awt.event.KeyEvent;

public class TextState implements State {
    @Override
    public void action(TextEditor context, KeyEvent e) {
        if(e.isActionKey()){
            context.setState(new ActionState());
            return;
        }
        char chr = e.getKeyChar();
        if(Character.isAlphabetic(chr) || Character.isDigit(chr) || Character.isSpaceChar(chr )){
            context.getModel().setSelectionRange(null);
            context.getModel().insert(chr);
        }
    }
}
