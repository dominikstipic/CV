package interfaces;

import textEditor.TextEditor;
import java.awt.event.KeyEvent;

public interface State {
    void action(TextEditor context, KeyEvent e);
}
