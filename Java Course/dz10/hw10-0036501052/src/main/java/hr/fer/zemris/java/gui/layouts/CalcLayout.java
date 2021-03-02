package hr.fer.zemris.java.gui.layouts;

import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Insets;
import java.awt.LayoutManager2;
import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;
import static hr.fer.zemris.java.gui.layouts.RCPosition.MAX_COLOMN;
import static hr.fer.zemris.java.gui.layouts.RCPosition.MAX_ROW;

/**
 * Represents LayOut manager for organizing components.
 * Components are organized in such manner that organization looks like 
 * calculator interface
 * @author Dominik Stipić
 *
 */
public class CalcLayout implements LayoutManager2 {
	/**
	 * length of gap beetween components
	 */
	private int gap;
	/**
	 * all components in this manager
	 */
	private Map<Component, RCPosition> componentIndex = new HashMap<>();
	/**
	 * Max function
	 */
	private BiFunction<Integer, Integer, Integer> MAX = (v1, v2) -> Math.max(v1, v2);
	/**
	 * Min function
	 */
	private BiFunction<Integer, Integer, Integer> MIN = (v1, v2) -> Math.min(v1, v2);

	/**
	 * Creates Layout manager with gap beetween components
	 * @param gap length
	 */
	public CalcLayout(int gap) {
		if (gap < 0) {
			throw new CalcLayoutException("gap beetween components cannot be negative");
		}
		this.gap = gap;
	}

	/**
	 * Creates Layout manager with no gap beetween components
	 */
	public CalcLayout() {
		this(0);
	}

	@Override
	public void removeLayoutComponent(Component comp) {
		componentIndex.remove(comp);
	}

	@Override
	public void layoutContainer(Container container) {
		Insets ins = container.getInsets();
		int containerWidth = container.getWidth()-ins.left-ins.right;
		int containerHeight = container.getHeight()-ins.bottom-ins.top;
		
		int componentHeight = ((containerHeight - 4*gap) / MAX_ROW);
		int componentWidth = ((containerWidth - 6*gap) / MAX_COLOMN);
		
		componentIndex.forEach((comp,rc)->{
			if(rc.equals(new RCPosition(1, 1))) {
				comp.setBounds(0, 0, componentWidth*5 + 4*gap, componentHeight);
			}
			else {
				int x = (rc.getColomn()-1) * componentWidth + (rc.getColomn()-1)*gap;
				int y = (rc.getRow()-1) * componentHeight + (rc.getRow()-1)*gap;
				comp.setBounds(x, y, componentWidth, componentHeight);
			}
		});
	}

	@Override
	public Dimension minimumLayoutSize(Container parent) {
		Insets ins = parent.getInsets();
		int preferredHeight = ins.top + ins.bottom;
		int preferredWidth = ins.left + ins.right;

		Dimension dim = determineDimensions(parent, Component::getMinimumSize, MIN);

		preferredWidth += (RCPosition.MAX_COLOMN) * dim.width + (RCPosition.MAX_COLOMN - 1) * gap;
		preferredHeight += RCPosition.MAX_ROW * dim.height + (RCPosition.MAX_ROW - 1) * gap;

		return new Dimension(preferredWidth, preferredHeight);
	}

	@Override
	public Dimension preferredLayoutSize(Container parent) {
		Insets ins = parent.getInsets();
		int preferredHeight = ins.top + ins.bottom;
		int preferredWidth = ins.left + ins.right;

		Dimension dim = determineDimensions(parent, Component::getPreferredSize, MAX);

		preferredWidth += (RCPosition.MAX_COLOMN) * dim.width + (RCPosition.MAX_COLOMN - 1) * gap;
		preferredHeight += RCPosition.MAX_ROW * dim.height + (RCPosition.MAX_ROW - 1) * gap;

		return new Dimension(preferredWidth, preferredHeight);
	}

	@Override
	public void addLayoutComponent(Component comp, Object obj) {
		String position;
		if (obj instanceof String) {
			position = (String) obj;
		} 
		else if(obj instanceof RCPosition) {
			RCPosition pos = (RCPosition) obj;
			position = pos.getRow() + "," + pos.getColomn();
		}
		else {
			throw new CalcLayoutException ("Constraint object is invalid");
		}
		checkPosition(position);
		String []parts = position.split(",");
		RCPosition rc = new RCPosition(Integer.valueOf(parts[0]), Integer.valueOf(parts[1]));
		if (componentIndex.containsKey(comp)) {
			throw new CalcLayoutException("Component has already been added");
		}
		if (componentIndex.containsValue(rc)) {
			throw new CalcLayoutException("Specified position is already occupied");
		}
		componentIndex.put(comp, rc);

	}

	@Override
	public float getLayoutAlignmentX(Container arg0) {
		return 0.5f;
	}

	@Override
	public float getLayoutAlignmentY(Container arg0) {
		return 0.5f;
	}

	@Override
	public Dimension maximumLayoutSize(Container parent) {
		Insets ins = parent.getInsets();
		int preferredHeight = ins.top + ins.bottom;
		int preferredWidth = ins.left + ins.right;

		Dimension dim = determineDimensions(parent, Component::getMaximumSize, MAX);

		preferredWidth += (RCPosition.MAX_COLOMN) * dim.width + (RCPosition.MAX_COLOMN - 1) * gap;
		preferredHeight += RCPosition.MAX_ROW * dim.height + (RCPosition.MAX_ROW - 1) * gap;

		return new Dimension(preferredWidth, preferredHeight);
	}

	/**
	 * Parses and checks position input 
	 * @param position string representation of layout position
	 */
	private void checkPosition(String position) {
		position = position.replaceAll("[ ]+", "");
		if (!position.matches("\\d,\\d")) {
			throw new CalcLayoutException("provided invalid position -> " + position);
		}
		String[] parts = position.split(",");
		if (parts[0].equals("1") && parts[1].matches("[2-5]")) {
			throw new CalcLayoutException(
					"provided invalid position.Positons (1,s) where s is from 2 to 5 are invalid");
		}
		if (!(parts[0].matches("[1-5]") && parts[1].matches("[1-7]"))) {
			throw new CalcLayoutException(
					"Number of raws cannot be bigger then 5 or negative or number of columns cannot be bigger then 7 or negative");
		}
	}

	/**
	 * generic method for determening prefferd,minimum or maximum dimensions of container
	 * @param parent container
	 * @param getDimension preffered,max or min
	 * @param math function for comparison
	 * @return calculated dimension
	 */
	private Dimension determineDimensions(Container parent, Function<Component, Dimension> getDimension,BiFunction<Integer, Integer, Integer> math) {
		Dimension dim = new Dimension(0, 0);
		for (Component c : parent.getComponents()) {
			Dimension d = getDimension.apply(c);
			if (d == null)continue;
			if (componentIndex.get(c).equals(new RCPosition(1, 1))) {
				d.width = (d.width - 4 * gap) / 5;
			}
			dim.height = math.apply(dim.height, d.height);
			dim.width = math.apply(dim.width, d.width);
		}
		return dim;
	}

	@Override
	public void addLayoutComponent(String position, Component comp) {}
	
	@Override
	public void invalidateLayout(Container arg0) {}

}
