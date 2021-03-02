package hr.fer.zemris.java.hw11.jnotepadpp;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

import javax.swing.AbstractAction;
import javax.swing.Action;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTextArea;
import javax.swing.JToolBar;
import javax.swing.KeyStroke;
import javax.swing.SwingUtilities;
import javax.swing.text.BadLocationException;
import javax.swing.text.Document;

public class JNotepadPP extends JFrame{
	private static final long serialVersionUID = -845616860954581153L;
	private JPanel cp = (JPanel) getContentPane();
	private JTabbedPane tabs = new JTabbedPane();
	private JToolBar toolBar = new JToolBar();
	private Map<Integer,Path>  filePaths = new HashMap<>();
	private Supplier<JTextArea> getTextArea = () -> (JTextArea) ((JScrollPane) tabs.getSelectedComponent()).getViewport().getComponents()[0];
	private DocumentListenerImpl docListener;
	private String buff;
	private ImageIcon green;
	private StatusBar statusBar;
	private CaseChanger changeCaser;
	

	/////////////////////////////////////////////////// ACTIONS
	
	private final Action addDocoument = new AbstractAction("add") {
		private static final long serialVersionUID = 1L;
		@Override
		public void actionPerformed(ActionEvent e) {
			String str = e.getActionCommand();
			if(str.equals("add")) {
				str = "new_" + tabs.getTabCount();
			}
			JTextArea area = new JTextArea();
			area.getDocument().addDocumentListener(docListener);
			area.addCaretListener(l -> {
				statusBar.update(l.getDot(), l.getMark());
				if(l.getDot() != l.getMark()) {
					changeCaser.activate();
				}
				else {
					changeCaser.unActivate();
				}
			});
			
			tabs.addTab(str, new JScrollPane(area));
			tabs.setIconAt(tabs.getSelectedIndex(), green);
			tabs.setSelectedComponent(tabs.getComponent(tabs.getComponentCount()-1));
		}
	}; 
	
	private final Action openDocument = new AbstractAction("open") {
		private static final long serialVersionUID = 1L;
		@Override
		public void actionPerformed(ActionEvent e) {
			JFileChooser fc = new JFileChooser();
			fc.setDialogTitle("Open file");
			if(fc.showOpenDialog(JNotepadPP.this)!=JFileChooser.APPROVE_OPTION) { 
				return;
			}
			Path path = fc.getSelectedFile().toPath();
			if(!Files.isReadable(path)) {
				JOptionPane.showMessageDialog(
						JNotepadPP.this,
						"File: " + path + " doesn't exist!",
						"Error",
						JOptionPane.ERROR_MESSAGE);
				return;
			}
			
			byte[] bytes;
			try {
				bytes = Files.readAllBytes(path);
			} catch(Exception ex) {
				JOptionPane.showMessageDialog(
						JNotepadPP.this,
						"Error while reading file from: " + path.toString(),
						"Error",
						JOptionPane.ERROR_MESSAGE);
				return;
			}
			String text = new String(bytes, StandardCharsets.UTF_8);
			
			ActionEvent event = new ActionEvent(this, 0, path.getFileName().toString());
			addDocoument.actionPerformed(event);
			getTextArea.get().setText(text);
			filePaths.put(tabs.getSelectedIndex(), path);
		}
	};
	
	
	private final Action saveAs = new AbstractAction("saveAs") {
		private static final long serialVersionUID = -1386361132117097406L;

		@Override
		public void actionPerformed(ActionEvent e) {
			JFileChooser fc = new JFileChooser();
			fc.setDialogTitle("save in directory");
			if(fc.showOpenDialog(JNotepadPP.this)!=JFileChooser.APPROVE_OPTION) { 
				return;
			}
			Path path = fc.getSelectedFile().toPath();
			
			if(Files.isRegularFile(path)) {
				if(JOptionPane.showConfirmDialog(
						JNotepadPP.this,
						"Do you want to overwritte existing file: " + path.toString(),
						"Warning",
						JOptionPane.WARNING_MESSAGE) != JOptionPane.OK_OPTION) {
					return;
				}
				try {
					Files.delete(path);
				} catch (IOException e1) {
					JOptionPane.showMessageDialog(
							JNotepadPP.this,
							"Error while overwitting file in: " + path.toString(),
							"Error",
							JOptionPane.ERROR_MESSAGE);
					return;
				}
			}
			
			filePaths.put(tabs.getSelectedIndex(), path);
			
			try (ByteArrayInputStream is = new ByteArrayInputStream(getTextArea.get().getText().getBytes())){
				Files.copy(is, path);
			} catch (IOException e1) {
				e1.printStackTrace();
				JOptionPane.showMessageDialog(
						JNotepadPP.this,
						"Error while saving file to: " + filePaths.get(tabs.getSelectedIndex()),
						"Error",
						JOptionPane.ERROR_MESSAGE);
				return;
			}
			tabs.setTitleAt(tabs.getSelectedIndex(), path.getFileName().toString());
			
			JOptionPane.showMessageDialog(
					JNotepadPP.this,
					"File saved",
					"info",
					JOptionPane.INFORMATION_MESSAGE);
			tabs.setIconAt(tabs.getSelectedIndex(), green);
		}
	};

	
	private final Action saveDocument = new AbstractAction("save") {
		private static final long serialVersionUID = 1L;
		@Override
		public void actionPerformed(ActionEvent e) {
			if(!filePaths.containsKey(tabs.getSelectedIndex())) {
				saveAs.actionPerformed(e);
				return;
			}
			try (ByteArrayInputStream is = new ByteArrayInputStream(getTextArea.get().getText().getBytes())){
				Files.delete(filePaths.get(tabs.getSelectedIndex()));
				Files.copy(is, filePaths.get(tabs.getSelectedIndex()));
			} catch (IOException e1) {
				e1.printStackTrace();
				JOptionPane.showMessageDialog(
						JNotepadPP.this,
						"Error while saving file to: " + filePaths.get(tabs.getSelectedIndex()),
						"Error",
						JOptionPane.ERROR_MESSAGE);
				return;
			}
			docListener.update(getTextArea.get().getDocument(), false);
			JOptionPane.showMessageDialog(
					JNotepadPP.this,
					"File saved",
					"info",
					JOptionPane.INFORMATION_MESSAGE);
			tabs.setIconAt(tabs.getSelectedIndex(), green);
		}
	};

	private final Action closeTab = new AbstractAction("close") {
		private static final long serialVersionUID = -5965976730132104066L;

		@Override
		public void actionPerformed(ActionEvent e) {
			Boolean value = docListener.getValue(getTextArea.get().getDocument());
			if(value == null)value = false;
			if(value) {
				if(JOptionPane.showConfirmDialog(
						JNotepadPP.this,
						"Do you want to save current file: " + filePaths.get(tabs.getSelectedIndex()),
						"Warning",
						JOptionPane.WARNING_MESSAGE) == JOptionPane.OK_OPTION) {
					saveDocument.actionPerformed(e);
				}
			}
			docListener.remove(getTextArea.get().getDocument());
			filePaths.remove(tabs.getSelectedIndex());	
			tabs.removeTabAt(tabs.getSelectedIndex());
		}
	};
	
	
	private final Action copy = new AbstractAction("copy") {
		private static final long serialVersionUID = -6041388601623379546L;

		@Override
		public void actionPerformed(ActionEvent e) {
			Document doc = getTextArea.get().getDocument();
			int len = Math.abs(
					getTextArea.get().getCaret().getDot()-getTextArea.get().getCaret().getMark());
			int offset = 0;
			if(len != 0) {
				offset = Math.min(getTextArea.get().getCaret().getDot(), getTextArea.get().getCaret().getMark());
			} else {
				len = doc.getLength();
			}
			try {
				buff = doc.getText(offset, len);
			} catch (BadLocationException ex) {
				ex.printStackTrace();
			}
			
		}
	};
	
	private final Action cut = new AbstractAction("cut") {
		private static final long serialVersionUID = -2299471860868571360L;

		@Override
		public void actionPerformed(ActionEvent e) {
			Document doc = getTextArea.get().getDocument();
			int len = Math.abs(
					getTextArea.get().getCaret().getDot()-getTextArea.get().getCaret().getMark());
			int offset = 0;
			if(len != 0) {
				offset = Math.min(getTextArea.get().getCaret().getDot(), getTextArea.get().getCaret().getMark());
			} else {
				len = doc.getLength();
			}
			try {
				buff = doc.getText(offset, len);
				doc.remove(offset, len);
			} catch (BadLocationException ex) {
				ex.printStackTrace();
			}
			
		}
	};
	
	private final Action paste = new AbstractAction("paste") {
		private static final long serialVersionUID = -2299471860868571360L;

		@Override
		public void actionPerformed(ActionEvent e) {
			if(buff == null) return;
			int offset = getTextArea.get().getCaretPosition();
			try {
				getTextArea.get().getDocument().insertString(offset, buff, null);
			} catch (BadLocationException e1) {
				e1.printStackTrace();
			}
		}
	};
	
	private final Action exit = new AbstractAction("exit") {
		private static final long serialVersionUID = 703318327819696642L;
		@Override
		public void actionPerformed(ActionEvent e) {
			dispose();
			System.exit(1);
		}
	};
	
	private final Action stats = new AbstractAction("stats") {
		private static final long serialVersionUID = 4737894038997212141L;

		@Override
		public void actionPerformed(ActionEvent e) {
			int chars = getTextArea.get().getText().length();
			int nonBlank = getTextArea.get().getText().replaceAll("[ ]+|[\n]+|[\t]+", "").length();
			int lines = getTextArea.get().getLineCount();
			
			String s = String.format("number of characters -> %d\nnumber of non-blank characters -> %d\nnumber of lines -> %d\n ", chars,nonBlank,lines);
			
			JOptionPane.showMessageDialog(JNotepadPP.this, s);
		}
	};
	
	/////////////////////////////////////////////////////////////////////////////
	
	
	@Override
	public void dispose() {
		for(int i = 0;i < tabs.getTabCount(); ++i) {
			JTextArea f = (JTextArea) ((JScrollPane) tabs.getComponentAt(i)).getViewport().getComponents()[0];
			Document d = f.getDocument();
			if(docListener.getValue(d) != null && docListener.getValue(d)) {
				if(JOptionPane.showConfirmDialog(
						JNotepadPP.this,
						"Do you want to save current file: " + tabs.getTitleAt(i),
						"Warning",
						JOptionPane.WARNING_MESSAGE) == JOptionPane.OK_OPTION) {
					saveDocument.actionPerformed(new ActionEvent(this, 0, "exit"));
				}
			}
		}
	}

	public JNotepadPP() {
		setSize(500, 500);
		setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
		setTitle("JNotepad++");
		addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				dispose();
				System.exit(1);
			}
		});
		green = loadIcon("icons/g.png");
		docListener = new DocumentListenerImpl(loadIcon("icons/r.png"),tabs);
		statusBar = new StatusBar(tabs);
		
		initGUI();
	}
	
	public void initGUI(){
		cp.add(tabs);
		cp.add(toolBar,BorderLayout.NORTH);
		addDocoument.actionPerformed(new ActionEvent(this, 0, "add"));;
		
		addTabs();
		addMenus();
		addShortCuts();
		
		cp.add(statusBar,BorderLayout.SOUTH);
		statusBar.update(0, 0);
	}
	
	private ImageIcon loadIcon(String path) {
		byte[] bytes = null;
		try(InputStream is = this.getClass().getResourceAsStream(path)){
			bytes = is.readAllBytes();
			if(bytes == null)throw new IllegalArgumentException();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
		return new ImageIcon(bytes);
	} 
	
	private void addTabs() {
		toolBar.add(new JButton(addDocoument));
		toolBar.add(new JButton(openDocument));
		toolBar.add(new JButton(saveDocument));
		toolBar.add(new JButton(saveAs));
		toolBar.add(new JButton(closeTab));
		toolBar.add(new JButton(copy));
		toolBar.add(new JButton(cut));
		toolBar.add(new JButton(paste));
		toolBar.add(new JButton(stats));
		toolBar.add(new JButton(exit));
	}
	
	private void addMenus() {
		JMenuBar menuBar = new JMenuBar();
		JMenu fileMenu = new JMenu("file");
		menuBar.add(fileMenu);
		JMenu editBar = new JMenu("edit");
		menuBar.add(editBar);
		changeCaser = new CaseChanger(tabs);
		menuBar.add(changeCaser);
		
		fileMenu.add(new JMenuItem(addDocoument));
		fileMenu.add(new JMenuItem(openDocument));
		fileMenu.addSeparator();
		fileMenu.add(new JMenuItem(saveDocument));
		fileMenu.add(new JMenuItem(saveAs));
		fileMenu.addSeparator();
		fileMenu.add(new JMenuItem(closeTab));
		fileMenu.add(new JMenuItem(exit));
		fileMenu.add(new JMenuItem(copy));
		fileMenu.add(new JMenuItem(cut));
		editBar.add(new JMenuItem(paste));
		editBar.add(new JMenuItem(stats));
		
		this.setJMenuBar(menuBar);
	}
	
	private void addShortCuts() {
		addDocoument.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control A"));
		openDocument.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control O"));
		saveDocument.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control S"));
		saveAs.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control alt S"));
		closeTab.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control shift C"));
		copy.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control C"));
		cut.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control X"));
		paste.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control V"));
		stats.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control I"));
		exit.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control E"));
	}
	
	public static void main(String[] args) {
		SwingUtilities.invokeLater(()->{
			new JNotepadPP().setVisible(true);
		});
	}
}
