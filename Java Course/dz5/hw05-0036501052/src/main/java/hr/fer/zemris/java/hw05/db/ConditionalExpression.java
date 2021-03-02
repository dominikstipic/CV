package hr.fer.zemris.java.hw05.db;

import java.util.Objects;

/**
 * Razred koji predstavlja jedan izraz dan upitom.Upit je sastavljen od:
 * atributa,operatora i literala.
 * Razred sadrzi objekte koji tijekom izvodenja provjeravaju logiku koja je dana s upitom.
 * @author Dominik Stipic
 *
 */
public class ConditionalExpression {
	/**
	 * Atribut koji se promatra
	 */
	private IFieldValueGetter field;
	/**
	 * operaor koji se izvodi nad atributom
	 */
	private IComparisonOperator operator;
	/**
	 * String s kojim se usporeduje atribut
	 */
	private String literal;
	
	/**
	 * Konstruktor za jedan upit
	 * @param field Atribut 
	 * @param operator Operator
	 * @param literal Literal
	 * @throws NullPointerException - ako je neka vrijednost null
	 */
	public ConditionalExpression(IFieldValueGetter field, IComparisonOperator operator, String literal) {
		this.field = Objects.requireNonNull(field, "IFieldValueGetter value can't be null");
		this.operator = Objects.requireNonNull(operator, "IComaparisonOperator value can't be null");
		this.literal = Objects.requireNonNull(literal, "String value can't be null");
	}

	/**
	 * Getter
	 * @return atribut
	 */
	public IFieldValueGetter getFieldGetter() {
		return field;
	}

	/**
	 * Getter
	 * @return operator
	 */
	public IComparisonOperator getComparisonOperator() {
		return operator;
	}

	/**
	 * Getter
	 * @return String literal
	 */
	public String getStringLiteral() {
		return literal;
	}

}
