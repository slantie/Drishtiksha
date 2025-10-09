// src/components/media/MediaSearchFilter.jsx

import React, { useState, useEffect, useMemo } from 'react';
import { Search, X, FileVideo, FileAudio, FileImage, ShieldCheck, ShieldAlert } from 'lucide-react';
import { Input } from '../ui/Input';
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '../ui/Select';
import { Button } from '../ui/Button';
import { Card, CardContent } from '../ui/Card';

export const MediaSearchFilter = ({ mediaItems, onFilterChange }) => {
    const [filters, setFilters] = useState({
        searchTerm: '',
        status: 'ALL',
        mediaType: 'ALL',
        prediction: 'ALL',
    });
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
    // Includes statuses and media types from actual data.
    const availableOptions = useMemo(() => {
        const statuses = new Set();
        const mediaTypes = new Set();
        const predictions = new Set(); // To dynamically show if REAL/FAKE predictions exist
        
        mediaItems.forEach(item => {
            if (item.status) statuses.add(item.status);
            if (item.mediaType) mediaTypes.add(item.mediaType);

            // Add prediction types if any analysis run is completed
            const latestRun = item.analysisRuns?.[0];
            if (latestRun && latestRun.status === "ANALYZED") {
                 const completedAnalyses = latestRun.analyses?.filter(a => a.status === "COMPLETED") || [];
                 const realCount = completedAnalyses.filter(a => a.prediction === "REAL").length;
                 const fakeCount = completedAnalyses.filter(a => a.prediction === "FAKE").length;

                 if (realCount > fakeCount) predictions.add("REAL");
                 else if (fakeCount > realCount) predictions.add("FAKE");
            }
        });
        return {
            statuses: Array.from(statuses).sort(),
            mediaTypes: Array.from(mediaTypes).sort(),
            predictions: Array.from(predictions).sort(),
        };
    }, [mediaItems]);

    const hasActiveFilters = Object.values(filters).some(v => v !== '' && v !== 'ALL');

    return (
        <Card>
            <CardContent className="p-4">
                <div className="flex gap-2 items-center justify-between">
                    {/* Search Input */}
                    <div className='w-1/2'>
                        <Input
                        type="text"
                        placeholder="Search by filename or description..."
                        value={filters.searchTerm}
                        onChange={(e) => handleFilterChange('searchTerm', e.target.value)}
                        leftIcon={<Search className="h-5 w-5" />}
                        className="rounded-full w-full"
                        rightIcon={<></>}
                    />
                    </div>

                    {/* Media Type Filter */}
                    <div className="flex gap-2 w-1/2">
                    <Select value={filters.mediaType} onValueChange={(value) => handleFilterChange('mediaType', value)} className="w-1/6">
                        <SelectTrigger><SelectValue placeholder="Filter by Type" /></SelectTrigger>
                        <SelectContent>
                            <SelectItem value="ALL">All Media Types</SelectItem>
                            {availableOptions.mediaTypes.map(type => (
                                <SelectItem key={type} value={type} className="capitalize">
                                    <div className="flex items-center gap-2">
                                        {type === 'VIDEO' && <FileVideo className="h-4 w-4 text-purple-500" />}
                                        {type === 'AUDIO' && <FileAudio className="h-4 w-4 text-blue-500" />}
                                        {type === 'IMAGE' && <FileImage className="h-4 w-4 text-green-500" />}
                                        <span>{type.toLowerCase()}</span>
                                    </div>
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                    
                    {/* Status Filter */}
                    <Select value={filters.status} onValueChange={(value) => handleFilterChange('status', value)}>
                        <SelectTrigger><SelectValue placeholder="Filter by Status" /></SelectTrigger>
                        <SelectContent>
                            <SelectItem value="ALL">All Statuses</SelectItem>
                            {/* Filter out 'UNKNOWN' if it's not a real status to show */}
                            {availableOptions.statuses.filter(s => s !== 'UNKNOWN').map(status => (
                                <SelectItem key={status} value={status} className="capitalize">
                                    {status.replace('_', ' ').toLowerCase()}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                    
                    {/* Prediction Filter (based on overall assessment of latest run) */}
                    <Select value={filters.prediction} onValueChange={(value) => handleFilterChange('prediction', value)}>
                        <SelectTrigger><SelectValue placeholder="Filter by Result" /></SelectTrigger>
                        <SelectContent>
                            <SelectItem value="ALL">All Results</SelectItem>
                            {availableOptions.predictions.includes("REAL") && (
                                <SelectItem value="REAL">
                                    <div className="flex items-center gap-2">
                                        <ShieldCheck className="h-4 w-4 text-green-500" />
                                        <span>Likely Authentic</span>
                                    </div>
                                </SelectItem>
                            )}
                            {availableOptions.predictions.includes("FAKE") && (
                                <SelectItem value="FAKE">
                                    <div className="flex items-center gap-2">
                                        <ShieldAlert className="h-4 w-4 text-red-500" />
                                        <span>Likely Deepfake</span>
                                    </div>
                                </SelectItem>
                            )}
                        </SelectContent>
                    </Select>
                    </div>

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