// src/utils/formatKeys.js

import _ from "lodash";

/**
 * Recursively converts object keys from snake_case to camelCase.
 * @param {any} data - The data (object, array, or primitive) to transform.
 * @returns {any} The transformed data with camelCase keys.
 */
export const toCamelCase = (data) => {
    if (_.isArray(data)) {
        return data.map((item) => toCamelCase(item));
    }
    if (_.isObject(data) && data !== null) {
        return _.reduce(
            data,
            (result, value, key) => {
                const camelKey = _.camelCase(key);
                result[camelKey] = toCamelCase(value);
                return result;
            },
            {}
        );
    }
    return data;
};
