package hr.fer.zemris.java.hw05.db;

/**
 * Suclje koje sluzi za uspoređuje dva String literela uz neku 
 * definiranu operaciju.
 * @author Dominik Stipic
 *
 */
public interface IComparisonOperator {
	/**
	 * Vrsi operaciju nad ulazinim nizovima i vraca korespodentnu istinitnosnu vrijednost za danu operaciju
	 * @param value1 String literal
	 * @param value2 String literal
	 * @return true - operacije je zadovoljena, false inace
	 */
	boolean satisfied(String value1, String value2);
}
