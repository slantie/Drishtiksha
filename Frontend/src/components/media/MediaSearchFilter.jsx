// src/components/media/MediaSearchFilter.jsx

import React, { useState, useEffect, useMemo } from 'react';
import { Search, X, Video, FileAudio, Image as ImageIcon } from 'lucide-react';
import { Input } from '../ui/Input';
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '../ui/Select';
import { Button } from '../ui/Button';
import { Card, CardContent } from '../ui/Card';

export const MediaSearchFilter = ({ mediaItems, onFilterChange }) => {
    // REFACTOR: The component now manages its entire state internally.
    const [filters, setFilters] = useState({
        searchTerm: '',
        status: 'ALL',
        mediaType: 'ALL',
        prediction: 'ALL',
    });

    // Notify the parent component whenever the filters change.
    useEffect(() => {
        onFilterChange(filters);
    }, [filters, onFilterChange]);

    const handleFilterChange = (key, value) => {
        setFilters(prev => ({ ...prev, [key]: value }));
    };

    const handleClearFilters = () => {
        setFilters({
            searchTerm: '',
            status: 'ALL',
            mediaType: 'ALL',
            prediction: 'ALL',
        });
    };
    
    // Dynamically derive available filter options from the media data.
    const availableOptions = useMemo(() => {
        const statuses = new Set();
        const mediaTypes = new Set();
        mediaItems.forEach(item => {
            if (item.status) statuses.add(item.status);
            if (item.mediaType) mediaTypes.add(item.mediaType);
        });
        return {
            statuses: Array.from(statuses).sort(),
            mediaTypes: Array.from(mediaTypes).sort(),
        };
    }, [mediaItems]);

    const hasActiveFilters = Object.values(filters).some(v => v !== '' && v !== 'ALL');

    return (
        <Card>
            <CardContent className="p-4">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 items-center">
                    {/* Search Input */}
                    <Input
                        type="text"
                        placeholder="Search by filename..."
                        value={filters.searchTerm}
                        onChange={(e) => handleFilterChange('searchTerm', e.target.value)}
                        leftIcon={<Search />}
                        className="lg:col-span-1"
                    />

                    {/* Media Type Filter */}
                    <Select value={filters.mediaType} onValueChange={(value) => handleFilterChange('mediaType', value)}>
                        <SelectTrigger><SelectValue placeholder="Filter by Type" /></SelectTrigger>
                        <SelectContent>
                            <SelectItem value="ALL">All Media Types</SelectItem>
                            {availableOptions.mediaTypes.map(type => (
                                <SelectItem key={type} value={type} className="capitalize">
                                    {type.toLowerCase()}s
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                    
                    {/* Status Filter */}
                    <Select value={filters.status} onValueChange={(value) => handleFilterChange('status', value)}>
                        <SelectTrigger><SelectValue placeholder="Filter by Status" /></SelectTrigger>
                        <SelectContent>
                            <SelectItem value="ALL">All Statuses</SelectItem>
                            {availableOptions.statuses.map(status => (
                                <SelectItem key={status} value={status} className="capitalize">
                                    {status.replace('_', ' ').toLowerCase()}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                    
                    {/* Prediction Filter */}
                    <Select value={filters.prediction} onValueChange={(value) => handleFilterChange('prediction', value)}>
                        <SelectTrigger><SelectValue placeholder="Filter by Result" /></SelectTrigger>
                        <SelectContent>
                            <SelectItem value="ALL">All Results</SelectItem>
                            <SelectItem value="REAL">Likely Authentic</SelectItem>
                            <SelectItem value="FAKE">Likely Deepfake</SelectItem>
                        </SelectContent>
                    </Select>

                </div>
                {hasActiveFilters && (
                    <div className="mt-3 flex justify-end">
                        <Button variant="ghost" onClick={handleClearFilters}>
                            <X className="mr-2 h-4 w-4" /> Clear All Filters
                        </Button>
                    </div>
                )}
            </CardContent>
        </Card>
    );
};