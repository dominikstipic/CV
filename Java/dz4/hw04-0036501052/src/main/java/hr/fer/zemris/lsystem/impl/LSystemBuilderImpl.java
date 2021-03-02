package hr.fer.zemris.lsystem.impl;

import static java.lang.Math.pow;

import java.awt.Color;
import java.util.Objects;

import hr.fer.zemris.java.custom.collections.Dictionary;
import hr.fer.zemris.lsystem.impl.commands.ColorCommand;
import hr.fer.zemris.lsystem.impl.commands.DrawCommand;
import hr.fer.zemris.lsystem.impl.commands.PopCommand;
import hr.fer.zemris.lsystem.impl.commands.PushCommand;
import hr.fer.zemris.lsystem.impl.commands.RotateCommand;
import hr.fer.zemris.lsystem.impl.commands.ScaleCommand;
import hr.fer.zemris.lsystem.impl.commands.SkipCommand;
import hr.fer.zemris.lsystems.LSystem;
import hr.fer.zemris.lsystems.LSystemBuilder;
import hr.fer.zemris.lsystems.Painter;
import hr.fer.zemris.math.Vector2D;

/**
 * Configures all properties needed for LSystem fractal creation.
 * User can configure fractal system by using class methods or by writing scripts 
 * and passing that script to configureFromText method.
 * @author Dominik Stipic
 *
 */
public class LSystemBuilderImpl implements LSystemBuilder {
	/**
	 * Dictionary storing pairs (symbol,action)
	 */
	private Dictionary actions;
	/**
	 * Dicitonary storing pairs (symbol, symbol production)
	 */
	private Dictionary production;

	/**
	 * length of turtle move
	 */
	private double unitLength;
	/**
	 * scaling factor for scaling unitLength
	 */
	private double unitLengthDegreeScaler;
	/**
	 * Origin point of turlte, from interval [0-1]x and [0-1]y
	 */
	private Vector2D origin;
	/**
	 * positive angle which turtle direction closes with positive axis
	 */
	private double angle;
	/**
	 * Starting symbol need for starting generation of new producted sequence
	 */
	private String axiom;

	/**
	 * Constructor which initializes fiels to default value:
	 * <pre>unitLength = 0.1;</pre>
	 * <pre>unitLengthDegreeScaler = 1;</pre>
	 * <pre>origin = new Vector2D(0, 0);</pre>
	 * <pre>angle = 0;</pre>
	 * <pre>axiom = ""</pre>
	 */
	public LSystemBuilderImpl() {
		actions = new Dictionary();
		production = new Dictionary();
		unitLength = 0.1;
		unitLengthDegreeScaler = 1;
		origin = new Vector2D(0, 0);
		angle = 0;
		axiom = "";

	}

	
	
	
	
	/**
	 * buildes LSystem from given configuration
	 * @return - builded LSystem
	 */
	@Override
	public LSystem build() {
		return new LSystemGenerator();
	}

	/** 
	 * Buildes LSystem configuratin form given script
	 * @param text - given script
	 * @return updated LSystem
	 * @throws NullPointerException - if text is null
	 */
	@Override
	public LSystemBuilder configureFromText(String[] text) {
		Objects.requireNonNull(text);

		for (String line : text) {
			if (line.trim().isEmpty())
				continue;
			translate(line);
		}
		
		return this;
	}

	/**
	 * Registers command and symbol for this command.
	 * @param symbol - symbol which is connected to command
	 * @param line - string representation of command
	 * @return updated LSystem
	 * @throws NullPointerException - if line or symobol is null
	 */
	@Override
	public LSystemBuilder registerCommand(char symbol, String line) {
		String sign = String.valueOf(symbol);

		Objects.requireNonNull(sign);
		Objects.requireNonNull(line);

		Command action = returnAction(line);
		actions.put(sign, action);
		return this;
	}

	/**
	 * Registers produciton from given symobol
	 * @param symbol - non terminating symobol which can be raplaced with its production equivalent
	 * @param sequence - sequence in which symobl can be tranformed
	 * @return updated LSystem
	 * @throws NullPointerException - if the arguments are null
	 */
	@Override
	public LSystemBuilder registerProduction(char symbol, String sequence) {
		String sign = String.valueOf(symbol);
		Objects.requireNonNull(sign);
		production.put(sign, sequence);
		return this;
	}

	/**
	 * Sets angle of turtle to specified angle
	 * @param angle - angle in degress 
	 * @return updated LSystem
	 */
	@Override
	public LSystemBuilder setAngle(double angle) {
		this.angle = angle;
		return this;
	}

	/**
	 * Sets initial axiom 
	 * @param axiom - initial starting axiom
	 * @return updated LSystem
	 * @throws NullPointerException - if axiom is null
	 */
	@Override
	public LSystemBuilder setAxiom(String axiom) {
		this.axiom = Objects.requireNonNull(axiom);
		return this;
	}

	/**
	 * Sets origin of turtle.Origin can be form interval [0-1] for x and [0-1] to y
	 * @param x - x coordinate
	 * @param y - y coordinate 
	 * @return updated LSystem
	 * @throws IllegalArgumentException - when origin isnt form defined interval
	 */
	@Override
	public LSystemBuilder setOrigin(double x, double y) {
		if(x < 0 && x > 1 && y < 0 && y > 1) {
			throw new IllegalArgumentException("Origin is out of bonds - must be from [0-1][0-1]");
		}
		origin = new Vector2D(x, y);
		return this;
	}

	/**
	 * Sets unit length of turtle. Unit length must be from [0,1]
	 * @param unitLength - one step of turtle 
	 * @return updated LSystem
	 * @throws IllegalArgumentException - if unit length isnt form defined interval
	 */
	@Override
	public LSystemBuilder setUnitLength(double unitLength) {
		if (unitLength < 0 || unitLength > 1) {
			throw new IllegalArgumentException("unit length must be from interval [0,1]");
		}
		this.unitLength = unitLength;
		return this;
	}

	/**
	 * Sets unit length degree scaler of turtle.Unit length degree scaler length must be from [0,1]
	 * @param unitLengthDegreeScaler - scaling factor for turtle step
	 * @return updated LSystem
	 * @throws IllegalArgumentException - if the unitLengthDegreeScaler isnt form defined interval 
	 */
	@Override
	public LSystemBuilder setUnitLengthDegreeScaler(double unitLengthDegreeScaler) {
		if (unitLengthDegreeScaler < 0 || unitLengthDegreeScaler > 1) {
			throw new IllegalArgumentException("unit length degree scaler must be from interval [0,1]");
		}
		this.unitLengthDegreeScaler = unitLengthDegreeScaler;
		return this;
	}

	
	/**
	 * Returns Command interface,which can be executed, from given command line
	 * @param str command line which is going to be interpreted
	 * @return corresponding command action from given legal command line
	 */
	private Command returnAction(String str) {
		String[] tokens = str.split("[ ]+");
		Command command;

		if ((tokens[0].equals("push") || tokens[0].equals("pop")) == false) {
			boolean isInt = false;
			if(tokens[0].equals("color")) isInt = true;
			double arg = checkArgument(tokens[1],isInt);
			command = createCommand(tokens[0], arg);
		} else {
			command = createCommand(tokens[0], -1); // -1 -> don't care(push,pop dont have args)
		}

		return command;
	}

	/**
	 * Creates command by checking if command exists and if so, it returnes created Command action
	 * @param str - keyword which represents command
	 * @param arg - argument for certain command
	 * @return created Command which can be executed
	 */
	private Command createCommand(String str, double arg) {
		if (str.equals("color")) {
			int colorNum = (int) arg;
			return new ColorCommand(new Color(colorNum));
		} else if (str.equals("draw")) {
			return new DrawCommand(arg);
		} else if (str.equals("skip")) {
			return new SkipCommand(arg);
		} else if (str.equals("scale")) {
			return new ScaleCommand(arg);
		} else if (str.equals("rotate")) {
			return new RotateCommand(arg);
		} else if (str.equals("push")) {
			return new PushCommand();
		} else if (str.equals("pop")) {
			return new PopCommand();
		} else {
			throw new IllegalArgumentException("command isn't defined - " + str);
		}
	}

	/**
	 * Checks if given argument can be parsed to integer or double and if so,it returnes
	 * parsed string
	 * @param arg which is going to be parsed 
	 * @param type -  true -> integer argument, false -> double argument
	 * @throws NumberFormatException - if string cannot be parsed to integer or double 
	 * @return parsed string
	 */
	private double checkArgument(String arg, boolean type) {
		try {
			double value;
			if(type) {
				value = Integer.parseInt(arg,16);
			}else {
				value = Double.parseDouble(arg);
			}
			return value;
		} catch (NumberFormatException e) {
			throw new NumberFormatException("Command expected argument but didn't find him");
		}

	}

	/**
	 * translates script lines giving it appropriate meaning.
	 * While translating it adjusts class field to stated configuration
	 * @param lines - script lines 
	 */
	private void translate(String lines) {
		String []tokens = lines.split("[ ]+");
		String str = tokens[0];
		int EXPECTED_SIZE;
		if(str.equals("origin")) {
			EXPECTED_SIZE = 3;
			translateChecker("origin", tokens, EXPECTED_SIZE, (args,size) -> {checkArgument(args[1], false);
			});
			
			double x0 = Double.valueOf(tokens[1]);
			double y0 = Double.valueOf(tokens[2]);
			setOrigin(x0, y0);
		}
		else if(str.equals("angle")) {
			EXPECTED_SIZE = 2;
			translateChecker("angle", tokens, EXPECTED_SIZE, (args,size) -> {checkArgument(args[1], false);
			});
			
			double angle = Double.valueOf(tokens[1]);
			setAngle(angle);
		}
		else if(str.equals("unitLength")) {
			EXPECTED_SIZE = 2;
			translateChecker("unitLength", tokens, EXPECTED_SIZE, (args,size) -> {checkArgument(args[1], false);
			});
			
			double unitLength = Double.valueOf(tokens[1]);
			setUnitLength(unitLength);
		}
		else if(str.equals("unitLengthDegreeScaler")) {
			String []tokens2 = lines.split("/");
			String []tokens1 = tokens2[0].split("[ ]+"); 
			if(tokens2.length != 1 && tokens1.length != 2) {
				throw new IllegalArgumentException("Error using unitLengthDegreeScaler command");
			}
			double num2 = checkArgument(tokens2[1].trim(), false);
			double num1 = checkArgument(tokens1[1], false);
			double unitLengthDegreeScaler = Double.valueOf(num1 / num2);
			this.unitLengthDegreeScaler = unitLengthDegreeScaler;
		}
		else if(str.equals("command")) {
			if(tokens.length != 4 && tokens.length != 3 ) {
				throw new IllegalArgumentException("Error using commands");
			}
			String commandLine = tokens[2] ;
			String symbol = tokens[1];
			if(tokens.length == 4) {
				commandLine += " " + tokens[3]; 
			} 
			Command action = returnAction(commandLine);
			actions.put(symbol, action);
		}
		else if(str.equals("axiom")) {
			EXPECTED_SIZE = 2;
			translateChecker("axiom", tokens, EXPECTED_SIZE, (args,size) -> setAxiom(tokens[1]));
		}
		else if(str.equals("production")) {
			EXPECTED_SIZE = 3;
			translateChecker("production", tokens, EXPECTED_SIZE, (args,size) -> {production.put(tokens[1], tokens[2]);
					});
		}
		else {
			throw new IllegalArgumentException("Unrecognizible Keyword - " + str);
		}
	}

	
	/**
	 * Checks if the given script line is correctly writen .  
	 * @param keyword - keyword used in this script language
	 * @param tokens - dismantlet script line to its token units 
	 * @param expectedSize - expected size for this script line, depends on keyword
	 * @param action which is going to be made,depends on keyword type
	 * @throws IllegalArgumentException - if the script isnt correctly writen
	 */
	private void translateChecker(String keyword, String[] tokens, int expectedSize, Action action) {
		if (tokens.length != expectedSize) {
			throw new IllegalArgumentException(keyword + " keyword expects " + (expectedSize - 1) + "arguments");
		}
		action.action(tokens, expectedSize);
	}

	/**
	 * Starts fractal creation process. Process is started when all 
	 * configuration data is correctly adjusted 
	 * @author Dominik Stipic
	 *
	 */
	private class LSystemGenerator implements LSystem {
		Context ctx;

		@Override
		public void draw(int level, Painter painter) {
			ctx = new Context();
			ctx.pushState(new TurtleState(origin, Vector2D.fromMagnitudeAndAngle(1, angle), Color.BLACK,
					unitLength * pow(unitLengthDegreeScaler, level)));

			String sequence = generate(level);
			int size = sequence.length();
			for (int i = 0; i < size; ++i) {
				String symbol = String.valueOf(sequence.charAt(i));
				Command command = (Command) actions.get(symbol);
				if (command == null) continue;
				command.execute(ctx, painter);
			}
			ctx.clearContext();

		}

		@Override
		public String generate(int level) {
			if (level == 0)
				return axiom;

			String sequence = axiom;
			for (int j = 0; j < level; ++j) {
				char[] arr = sequence.toCharArray();
				sequence = "";
				for (int i = 0; i < arr.length; ++i) {
					String derived = (String) production.get(String.valueOf(arr[i]));
					if (derived == null) {
						sequence += String.valueOf(arr[i]);
					} else {
						sequence += derived;
					}
				}
			}
			return sequence;
		}
	}
}
