package hr.fer.zemris.java.hw05.db;

import java.util.List;
import java.util.Objects;

/**
 * Implementacija generickog filtra koji filtrira studente prema danom upitu.
 * @author Dominik Stipic
 *
 */
public class QueryFilter implements IFilter{
	/**
	 * interna lista u koju se spremaju izrazi koji ce biti filtrirani
	 */
	private List<ConditionalExpression> list;
	
	/**
	 * @param list lista studenta koja ce biti selektirana
	 */
	public QueryFilter(List<ConditionalExpression> list) {
		this.list = Objects.requireNonNull(list, "Queries cannot be null");
	}
	
	@Override
	public boolean accepts(StudentRecord record) {
		for(ConditionalExpression expr : list) {
			IComparisonOperator operator = expr.getComparisonOperator();
			IFieldValueGetter fieldGetter = expr.getFieldGetter();
			
			if(!operator.satisfied(fieldGetter.get(record), expr.getStringLiteral())) {
				return false;
			}
		}
		return true;
	}
	
}
