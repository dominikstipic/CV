package hr.fer.zemris.java.hw05.db;

/**
 * Razred koji nudi staticke razrede koji izvode neku logicku operaciju nad String vrijedostima
 * @author Dominik Stipic
 *
 */
public class ComparisonOperators  {
	/**
	 *Provjerava dali je String v2 manje leksicke vrijednosti od v1
	 */
	public static final IComparisonOperator LESS = ((v1,v2) -> v1.compareTo(v2) < 0 ? true : false);
	/**
	 * Provjerava dali je String v2 manje ili jednke leksicke vrijednosti  od v1
	 */
	public static final IComparisonOperator LESS_OR_EQUALS = ((v1,v2) -> v1.compareTo(v2) <= 0 ? true : false);
	/**
	 * Provjerava dali je String v2 vece leksicke vrijednosti od v1
	 */
	public static final IComparisonOperator GREATER = ((v1,v2) -> v1.compareTo(v2) > 0 ? true : false);
	/**
	 * Provjerava dali je String v2 vece ili jedneke leksicke vrijednosti od v1
	 */
	public static final IComparisonOperator GREATER_OR_EQUALS  = ((v1,v2) -> v1.compareTo(v2) >= 0 ? true : false);
	/**
	 * Provjerava dali su Stringovi v1 i v2 jednaki
	 */
	public static final IComparisonOperator EQUALS = ((v1,v2) -> v1.compareTo(v2) == 0 ? true : false);
	/**
	 * Provjerava dali Stringovi v1 i v2 nisu jednaki
	 */
	public static final IComparisonOperator NOT_EQUALS = ((v1,v2) -> v1.compareTo(v2) == 0 ? false : true); 
	
	/**
	 * Provjeva dali dani String v1 zadovoljava uzorak dan s Stringom v2
	 */
	public static final IComparisonOperator LIKE = new IComparisonOperator() {
		@Override
		public boolean satisfied(String value1, String value2) {
			int numOfWildcasts = (int)value2.chars().filter(c -> c == '*' ).count();
			if(numOfWildcasts > 1) {
				throw new IllegalArgumentException("Using more then one wildcast isn't allowed");
			}
			
			if(numOfWildcasts == 1) {
				
				if(value2.matches(".+\\*.+")) {
					String parts [] = value2.split("\\*");
					return value1.matches(parts[0]+".+"+parts[1]);
				}
				else if(value2.matches("\\*.+")) {
					String str = value2.replaceAll("\\*", "");
					return value1.matches(".+"+str);
				}
				else if (value2.matches(".+\\*")){
					String str = value2.replaceAll("\\*", "");
					return value1.matches(str+".+");
				}
				else {
					throw new IllegalArgumentException("Error using wildcard \"*\" - whitespaces aren't allowed between wildcard and word");
				}
			}
			else {
				return value1.equals(value2);
			}
		}
	}; 
}
