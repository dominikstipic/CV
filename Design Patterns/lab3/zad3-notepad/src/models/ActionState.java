package models;

import interfaces.State;
import textEditor.TextEditor;

import java.awt.event.KeyEvent;

public class ActionState implements State {
    @Override
    public void action(TextEditor context, KeyEvent e) {
        context.setState(new TextState());
    }
}
